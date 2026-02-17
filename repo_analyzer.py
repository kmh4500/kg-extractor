"""Mine the HF Transformers repo for model data, commit history, and documentation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from git import Repo, InvalidGitRepositoryError

from kg_extractor.models import RepoAnalysis

logger = logging.getLogger(__name__)

# Keywords indicating important architectural commits
EVOLUTION_KEYWORDS = [
    "add model", "new model", "flash attention", "quantization",
    "gradient checkpointing", "mixed precision", "kv cache",
    "rotary embedding", "rope", "grouped query attention", "gqa",
    "sliding window", "sparse attention", "lora", "peft",
    "token merging", "speculative decoding", "moe", "mixture of experts",
]

# Maximum number of commits to scan for evolution keywords
MAX_COMMIT_SCAN = 5000


class RepoAnalyzer:
    """Analyzes the HF Transformers git repository."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        try:
            self.repo = Repo(self.repo_path)
        except InvalidGitRepositoryError:
            raise ValueError(f"{repo_path} is not a valid git repository")

    def analyze(self) -> RepoAnalysis:
        """Run full analysis pipeline."""
        logger.info("Starting repo analysis of %s", self.repo_path)

        models = self._scan_models()
        logger.info("Found %d models", len(models))

        components = self._scan_shared_components()
        logger.info("Found %d shared components", len(components))

        key_commits = self._scan_evolution_commits()
        logger.info("Found %d key commits", len(key_commits))

        doc_summaries = self._scan_documentation()
        logger.info("Found %d model docs", len(doc_summaries))

        return RepoAnalysis(
            models=models,
            components=components,
            key_commits=key_commits,
            doc_summaries=doc_summaries,
        )

    def _scan_models(self) -> list[dict]:
        """Scan src/transformers/models/*/ for model directories."""
        models_dir = self.repo_path / "src" / "transformers" / "models"
        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return []

        models = []
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue

            model_info = {
                "name": model_dir.name,
                "path": str(model_dir.relative_to(self.repo_path)),
                "classes": [],
                "first_commit_date": None,
                "has_modeling": False,
                "has_config": False,
                "has_tokenizer": False,
            }

            # Check for key files
            for f in model_dir.iterdir():
                if f.name.startswith("modeling_"):
                    model_info["has_modeling"] = True
                    model_info["classes"].extend(self._extract_class_names(f))
                elif f.name.startswith("configuration_"):
                    model_info["has_config"] = True
                elif "tokeniz" in f.name:
                    model_info["has_tokenizer"] = True

            # Get first commit date for the model directory
            model_info["first_commit_date"] = self._get_first_commit_date(
                str(model_dir.relative_to(self.repo_path))
            )

            models.append(model_info)

        return models

    def _extract_class_names(self, filepath: Path) -> list[str]:
        """Extract class names from a Python file."""
        classes = []
        try:
            content = filepath.read_text(errors="replace")
            for match in re.finditer(r"^class\s+(\w+)\s*[\(:]", content, re.MULTILINE):
                classes.append(match.group(1))
        except Exception as e:
            logger.debug("Could not read %s: %s", filepath, e)
        return classes

    def _get_first_commit_date(self, path: str) -> Optional[str]:
        """Get the date of the first commit that touched a path."""
        try:
            commits = list(self.repo.iter_commits(paths=path, max_count=1, reverse=True))
            if commits:
                return commits[0].committed_datetime.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug("Could not get first commit for %s: %s", path, e)
        return None

    def _scan_shared_components(self) -> list[dict]:
        """Identify shared components across models."""
        components = []
        modeling_utils = self.repo_path / "src" / "transformers" / "modeling_utils.py"
        if modeling_utils.exists():
            classes = self._extract_class_names(modeling_utils)
            for cls_name in classes:
                components.append({
                    "name": cls_name,
                    "file": "src/transformers/modeling_utils.py",
                    "type": "shared_base",
                })

        # Check for attention implementations
        attn_dir = self.repo_path / "src" / "transformers" / "models"
        if attn_dir.exists():
            attn_classes = set()
            for modeling_file in attn_dir.rglob("modeling_*.py"):
                try:
                    content = modeling_file.read_text(errors="replace")
                    for match in re.finditer(
                        r"^class\s+(\w*(?:Attention|SelfAttention|MultiHeadAttention)\w*)\s*[\(:]",
                        content,
                        re.MULTILINE,
                    ):
                        attn_classes.add(match.group(1))
                except Exception:
                    continue
            # Report a summary rather than every individual class
            if attn_classes:
                components.append({
                    "name": "attention_implementations",
                    "count": len(attn_classes),
                    "examples": sorted(attn_classes)[:10],
                    "type": "attention_variants",
                })

        return components

    def _scan_evolution_commits(self) -> list[dict]:
        """Scan commit history for key architectural evolution commits."""
        key_commits = []
        try:
            for i, commit in enumerate(self.repo.iter_commits(max_count=MAX_COMMIT_SCAN)):
                msg_lower = commit.message.lower()
                for keyword in EVOLUTION_KEYWORDS:
                    if keyword in msg_lower:
                        key_commits.append({
                            "sha": commit.hexsha[:8],
                            "date": commit.committed_datetime.strftime("%Y-%m-%d"),
                            "message": commit.message.strip().split("\n")[0][:200],
                            "keyword": keyword,
                            "author": str(commit.author),
                        })
                        break  # one keyword match per commit is enough
        except Exception as e:
            logger.warning("Error scanning commits: %s", e)
        return key_commits

    def _scan_documentation(self) -> list[dict]:
        """Extract summaries from model documentation."""
        docs_dir = self.repo_path / "docs" / "source" / "en" / "model_doc"
        if not docs_dir.exists():
            logger.warning("Docs directory not found: %s", docs_dir)
            return []

        summaries = []
        for doc_file in sorted(docs_dir.glob("*.md")):
            try:
                content = doc_file.read_text(errors="replace")
                # Extract first paragraph after the title as summary
                summary = self._extract_doc_summary(content)
                summaries.append({
                    "model": doc_file.stem,
                    "file": str(doc_file.relative_to(self.repo_path)),
                    "summary": summary,
                    "length": len(content),
                })
            except Exception as e:
                logger.debug("Could not read doc %s: %s", doc_file, e)

        return summaries

    def _extract_doc_summary(self, content: str) -> str:
        """Extract the first meaningful paragraph from a markdown doc."""
        lines = content.split("\n")
        in_content = False
        summary_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip frontmatter, comments, and headers
            if stripped.startswith("<!--") or stripped.startswith("---"):
                continue
            if stripped.startswith("#"):
                if in_content:
                    break  # hit next section
                in_content = True
                continue
            if in_content and stripped:
                summary_lines.append(stripped)
                if len(" ".join(summary_lines)) > 500:
                    break
            elif in_content and not stripped and summary_lines:
                break  # end of first paragraph

        return " ".join(summary_lines)[:500]
