"""Extract concepts and relationships from repo analysis using an LLM."""

from __future__ import annotations

import logging
from typing import Optional

from kg_extractor.graph import KnowledgeGraph
from kg_extractor.llm_client import get_client, chat_completion, parse_json_response
from kg_extractor.models import (
    ConceptNode, ConceptType, ConceptLevel, Edge, RelationshipType, RepoAnalysis,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are a transformer architecture expert. Given analysis data from the HuggingFace \
Transformers repository, extract a knowledge graph of concepts and their relationships.

For each concept, provide:
- id: snake_case identifier
- name: human-readable name
- type: one of {types}
- level: one of {levels}
- description: 1-2 sentence description
- key_ideas: list of 2-4 key ideas
- code_refs: list of relevant file:class references from the repo
- paper_ref: the seminal paper, e.g. "Vaswani et al., 2017 — Attention Is All You Need"

For each relationship (edge), provide:
- source: concept id
- target: concept id
- relationship: one of {relationships}
- description: brief description of the relationship

Focus on:
1. Core transformer concepts (attention, positional encoding, layer norm, etc.)
2. Model architectures (BERT, GPT, T5, LLaMA, etc.)
3. Key techniques (flash attention, quantization, KV cache, etc.)
4. Training innovations (pre-training objectives, RLHF, etc.)
5. Prerequisite chains (what must you understand before what)

Return ONLY valid JSON with keys "nodes" and "edges". No other text.\
"""

EXTRACTION_USER_PROMPT = """\
Here is the analysis of the HuggingFace Transformers repository:

## Models found ({num_models} total, showing first {shown_models}):
{models_text}

## Shared components:
{components_text}

## Key architectural commits ({num_commits} total, showing first {shown_commits}):
{commits_text}

## Documentation summaries ({num_docs} total, showing first {shown_docs}):
{docs_text}

Extract a comprehensive knowledge graph of transformer concepts and their \
relationships. Include foundational concepts (attention, embeddings) through \
frontier concepts (latest models). Ensure proper prerequisite chains.

Return ONLY valid JSON with keys "nodes" and "edges".\
"""


class ConceptExtractor:
    """Uses an LLM to extract concepts from repo analysis data."""

    def __init__(self, base_url: Optional[str] = None, model: str = "google/gemma-3-27b-it"):
        self.client = get_client(base_url)
        self.model = model

    def extract(self, analysis: RepoAnalysis) -> KnowledgeGraph:
        """Extract a knowledge graph from repo analysis."""
        logger.info("Extracting concepts via LLM (model=%s)", self.model)

        system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
            types=", ".join(t.value for t in ConceptType),
            levels=", ".join(l.value for l in ConceptLevel),
            relationships=", ".join(r.value for r in RelationshipType),
        )

        user_prompt = self._build_user_prompt(analysis)

        response_text = chat_completion(
            self.client, self.model, system_prompt, user_prompt,
            max_tokens=8192, temperature=0.3,
        )

        graph_data = parse_json_response(response_text)
        if not graph_data.get("nodes"):
            logger.warning("No nodes in response, retrying with simpler prompt...")
            graph_data = self._retry_extraction(system_prompt, analysis)

        return self._build_graph(graph_data)

    def _retry_extraction(self, system_prompt: str, analysis: RepoAnalysis) -> dict:
        """Retry with a shorter prompt if the first attempt fails."""
        short_prompt = (
            "Extract 30-40 key transformer concepts from the HuggingFace Transformers repo. "
            f"Models include: {', '.join(m['name'] for m in analysis.models[:30])}. "
            "Return ONLY valid JSON with keys 'nodes' and 'edges'."
        )
        text = chat_completion(
            self.client, self.model, system_prompt, short_prompt,
            max_tokens=8192, temperature=0.3,
        )
        return parse_json_response(text)

    def _build_user_prompt(self, analysis: RepoAnalysis) -> str:
        max_models = 50
        models_text = ""
        for m in analysis.models[:max_models]:
            date_str = m.get("first_commit_date", "unknown")
            classes = ", ".join(m.get("classes", [])[:5])
            models_text += f"- **{m['name']}** (first commit: {date_str}): {classes}\n"

        components_text = ""
        for c in analysis.components:
            if c.get("type") == "attention_variants":
                components_text += (
                    f"- {c['name']}: {c['count']} variants, "
                    f"e.g. {', '.join(c.get('examples', [])[:5])}\n"
                )
            else:
                components_text += f"- {c['name']} ({c.get('file', '')})\n"

        max_commits = 40
        commits_text = ""
        for c in analysis.key_commits[:max_commits]:
            commits_text += f"- [{c['date']}] {c['message']} (keyword: {c['keyword']})\n"

        max_docs = 40
        docs_text = ""
        for d in analysis.doc_summaries[:max_docs]:
            summary = d.get("summary", "")[:200]
            docs_text += f"- **{d['model']}**: {summary}\n"

        return EXTRACTION_USER_PROMPT.format(
            num_models=len(analysis.models),
            shown_models=min(len(analysis.models), max_models),
            models_text=models_text or "(none found)",
            components_text=components_text or "(none found)",
            num_commits=len(analysis.key_commits),
            shown_commits=min(len(analysis.key_commits), max_commits),
            commits_text=commits_text or "(none found)",
            num_docs=len(analysis.doc_summaries),
            shown_docs=min(len(analysis.doc_summaries), max_docs),
            docs_text=docs_text or "(none found)",
        )

    def _build_graph(self, data: dict) -> KnowledgeGraph:
        """Build a KnowledgeGraph from parsed extraction data."""
        kg = KnowledgeGraph()

        for node_data in data.get("nodes", []):
            try:
                node = ConceptNode(
                    id=node_data["id"],
                    name=node_data["name"],
                    type=ConceptType(node_data.get("type", "theory")),
                    level=ConceptLevel(node_data.get("level", "intermediate")),
                    description=node_data.get("description", ""),
                    key_ideas=node_data.get("key_ideas", []),
                    code_refs=node_data.get("code_refs", []),
                    paper_ref=node_data.get("paper_ref", ""),
                    first_appeared=node_data.get("first_appeared"),
                    confidence=node_data.get("confidence", 1.0),
                )
                kg.add_concept(node)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid node %s: %s", node_data.get("id", "?"), e)

        valid_ids = {n.id for n in kg.get_all_concepts()}
        for edge_data in data.get("edges", []):
            try:
                source = edge_data["source"]
                target = edge_data["target"]
                if source not in valid_ids or target not in valid_ids:
                    logger.debug("Skipping edge %s->%s: missing node", source, target)
                    continue
                edge = Edge(
                    source=source,
                    target=target,
                    relationship=RelationshipType(edge_data.get("relationship", "builds_on")),
                    description=edge_data.get("description", ""),
                )
                kg.add_edge(edge)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid edge: %s", e)

        logger.info(
            "Built knowledge graph: %d concepts, %d edges",
            len(kg.get_all_concepts()),
            len(kg.get_all_edges()),
        )
        return kg
