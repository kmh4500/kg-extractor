"""CLI entry point for the knowledge graph extractor."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from git import Repo as GitRepo

from kg_extractor.concept_extractor import ConceptExtractor
from kg_extractor.course_builder import CourseBuilder
from kg_extractor.expander import GraphExpander
from kg_extractor.graph import KnowledgeGraph
from kg_extractor.models import RepoAnalysis
from kg_extractor.repo_analyzer import RepoAnalyzer
from kg_extractor.scaffold import Scaffolder

logger = logging.getLogger("kg_extractor")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline: analyze → extract → expand → build courses → scaffold."""
    setup_logging(args.verbose)

    repo_path = Path(args.repo)

    # If it's a URL, clone it first
    if args.repo.startswith("http://") or args.repo.startswith("https://"):
        clone_dir = Path(args.clone_dir) if args.clone_dir else Path(tempfile.mkdtemp(prefix="kg_"))
        if not (clone_dir / ".git").exists():
            logger.info("Cloning %s to %s ...", args.repo, clone_dir)
            GitRepo.clone_from(args.repo, str(clone_dir), depth=1)
            # For commit history, do a partial unshallow
            logger.info("Fetching commit history (last %d commits)...", args.max_commits)
            git_repo = GitRepo(clone_dir)
            try:
                git_repo.git.fetch("--deepen", str(args.max_commits))
            except Exception as e:
                logger.warning("Could not deepen clone: %s", e)
        else:
            logger.info("Using existing clone at %s", clone_dir)
        repo_path = clone_dir

    output_dir = Path(args.output)

    # Phase 1: Repo analysis
    logger.info("=== Phase 1: Repo Analysis ===")
    analyzer = RepoAnalyzer(repo_path)
    analysis = analyzer.analyze()

    # Save intermediate analysis
    analysis_path = output_dir / "knowledge" / "repo_analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis.to_dict(), indent=2) + "\n")
    logger.info("Saved repo analysis to %s", analysis_path)

    # Phase 2: Concept extraction
    logger.info("=== Phase 2: Concept Extraction ===")
    extractor = ConceptExtractor(model=args.model)
    kg = extractor.extract(analysis)
    logger.info("Extracted %d concepts", len(kg.get_all_concepts()))

    # Phase 3: Graph expansion
    if not args.skip_expansion:
        logger.info("=== Phase 3: Graph Expansion ===")
        expander = GraphExpander(model=args.model)
        kg = expander.expand(kg, rounds=args.expansion_rounds)
        logger.info("Graph now has %d concepts after expansion", len(kg.get_all_concepts()))
    else:
        logger.info("=== Phase 3: Skipping graph expansion ===")

    # Phase 4: Course building
    logger.info("=== Phase 4: Course Building ===")
    builder = CourseBuilder(model=args.model)
    courses = builder.build_courses(kg, generate_lessons=not args.skip_lessons)
    logger.info("Built %d courses", len(courses))

    # Phase 5: Scaffold
    logger.info("=== Phase 5: Scaffolding Course Repo ===")
    scaffolder = Scaffolder(kg, courses, enable_blockchain=args.enable_blockchain)
    scaffolder.scaffold(output_dir, repo_path=repo_path)

    # Initialize git repo if needed
    if not (output_dir / ".git").exists():
        logger.info("Initializing git repo in %s", output_dir)
        GitRepo.init(str(output_dir))

    logger.info("Done! Course repo is at: %s", output_dir)
    logger.info("To start learning: cd %s && claude", output_dir)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run only the repo analysis phase."""
    setup_logging(args.verbose)
    analyzer = RepoAnalyzer(args.repo)
    analysis = analyzer.analyze()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis.to_dict(), indent=2) + "\n")
    logger.info("Analysis saved to %s", output)


def cmd_extract(args: argparse.Namespace) -> None:
    """Run concept extraction from an existing analysis file."""
    setup_logging(args.verbose)
    analysis_data = json.loads(Path(args.analysis).read_text())
    analysis = RepoAnalysis.from_dict(analysis_data)

    extractor = ConceptExtractor(model=args.model)
    kg = extractor.extract(analysis)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kg.save(output)
    logger.info("Graph saved to %s (%d concepts)", output, len(kg.get_all_concepts()))


def cmd_build(args: argparse.Namespace) -> None:
    """Build courses from an existing graph file."""
    setup_logging(args.verbose)
    kg = KnowledgeGraph.load(Path(args.graph))

    builder = CourseBuilder(model=args.model)
    courses = builder.build_courses(kg, generate_lessons=not args.skip_lessons)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    courses_data = [c.to_dict() for c in courses]
    output.write_text(json.dumps(courses_data, indent=2, ensure_ascii=False) + "\n")
    logger.info("Courses saved to %s (%d courses)", output, len(courses))


def cmd_scaffold(args: argparse.Namespace) -> None:
    """Generate a course repo from existing graph and courses files."""
    setup_logging(args.verbose)
    kg = KnowledgeGraph.load(Path(args.graph))

    from kg_extractor.models import Course
    courses_data = json.loads(Path(args.courses).read_text())
    courses = [Course.from_dict(c) for c in courses_data]

    scaffolder = Scaffolder(kg, courses, enable_blockchain=args.enable_blockchain)
    scaffolder.scaffold(args.output, repo_path=args.repo if args.repo else None)
    logger.info("Course repo scaffolded at %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kg_extractor",
        description="Extract knowledge graphs from code repos and build learning courses",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command (full end-to-end)
    p_pipeline = subparsers.add_parser("pipeline", help="Run full extraction pipeline")
    p_pipeline.add_argument("--repo", required=True, help="Path or URL to the HF transformers repo")
    p_pipeline.add_argument("--output", required=True, help="Output directory for the course repo")
    p_pipeline.add_argument("--clone-dir", default=None, help="Directory to clone repo into (if URL)")
    p_pipeline.add_argument("--model", default="google/gemma-3-27b-it", help="LLM model name on vLLM server")
    p_pipeline.add_argument("--max-commits", type=int, default=2000, help="Max commits to fetch for analysis")
    p_pipeline.add_argument("--expansion-rounds", type=int, default=2, help="Number of graph expansion rounds")
    p_pipeline.add_argument("--skip-expansion", action="store_true", help="Skip graph expansion phase")
    p_pipeline.add_argument("--skip-lessons", action="store_true", help="Skip lesson generation (faster)")
    p_pipeline.add_argument("--enable-blockchain", action="store_true", help="Generate blockchain/ directory with AIN helper")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # Analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze a repo (Phase 1 only)")
    p_analyze.add_argument("--repo", required=True, help="Path to the repo")
    p_analyze.add_argument("--output", default="analysis.json", help="Output file")
    p_analyze.set_defaults(func=cmd_analyze)

    # Extract command
    p_extract = subparsers.add_parser("extract", help="Extract concepts from analysis (Phase 2)")
    p_extract.add_argument("--analysis", required=True, help="Path to analysis.json")
    p_extract.add_argument("--output", default="graph.json", help="Output graph file")
    p_extract.add_argument("--model", default="google/gemma-3-27b-it", help="LLM model name on vLLM server")
    p_extract.set_defaults(func=cmd_extract)

    # Build command
    p_build = subparsers.add_parser("build", help="Build courses from graph (Phase 4)")
    p_build.add_argument("--graph", required=True, help="Path to graph.json")
    p_build.add_argument("--output", default="courses.json", help="Output courses file")
    p_build.add_argument("--model", default="google/gemma-3-27b-it", help="LLM model name on vLLM server")
    p_build.add_argument("--skip-lessons", action="store_true", help="Skip lesson generation")
    p_build.set_defaults(func=cmd_build)

    # Scaffold command
    p_scaffold = subparsers.add_parser("scaffold", help="Generate course repo (Phase 5)")
    p_scaffold.add_argument("--graph", required=True, help="Path to graph.json")
    p_scaffold.add_argument("--courses", required=True, help="Path to courses.json")
    p_scaffold.add_argument("--output", required=True, help="Output directory")
    p_scaffold.add_argument("--repo", default=None, help="Source repo path for code snippets")
    p_scaffold.add_argument("--enable-blockchain", action="store_true", help="Generate blockchain/ directory with AIN helper")
    p_scaffold.set_defaults(func=cmd_scaffold)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
