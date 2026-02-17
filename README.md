# kg-extractor

Extract knowledge graphs from source code repositories and generate interactive learning courses for [Claude Code](https://claude.ai/claude-code).

## What it does

```
Source Repo → Knowledge Graph → Courses & Lessons → Scaffolded Course Repo
```

1. **Analyze** a code repository (file structure, imports, class hierarchy, commit history)
2. **Extract** concepts and relationships using an LLM → knowledge graph
3. **Expand** the graph with frontier concepts over multiple rounds
4. **Build** courses with ordered lessons, exercises, and paper references
5. **Scaffold** a learner-facing course repo that works as a Claude Code project

The generated course repo includes a `CLAUDE.md` that turns Claude into an interactive tutor — it teaches concepts, tracks progress, and quizzes the learner.

## Requirements

- Python 3.10+
- A vLLM-compatible server running an LLM (default: `google/gemma-3-27b-it`)
- GitPython (`pip install gitpython`)
- Node.js >= 16 (only if using `--enable-blockchain`)

## Usage

### Full pipeline (recommended)

```bash
python3 -m kg_extractor pipeline \
  --repo /path/to/source-repo \
  --output /path/to/course-output
```

This runs all 5 phases and produces a ready-to-use course directory.

### With blockchain integration

```bash
python3 -m kg_extractor pipeline \
  --repo /path/to/source-repo \
  --output /path/to/course-output \
  --enable-blockchain
```

This adds on-chain progress tracking via the [AIN blockchain](https://ainetwork.ai/). Learners get a wallet, completions are recorded on-chain, and explorers can discover each other's progress.

### Individual phases

Run phases separately for more control:

```bash
# Phase 1: Analyze repo
python3 -m kg_extractor analyze --repo /path/to/repo --output analysis.json

# Phase 2: Extract concepts
python3 -m kg_extractor extract --analysis analysis.json --output graph.json

# Phase 4: Build courses
python3 -m kg_extractor build --graph graph.json --output courses.json

# Phase 5: Scaffold course repo
python3 -m kg_extractor scaffold --graph graph.json --courses courses.json --output /path/to/output
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | (required) | Path or URL to the source repository |
| `--output` | (required) | Output directory for the course repo |
| `--model` | `google/gemma-3-27b-it` | LLM model name on vLLM server |
| `--expansion-rounds` | `2` | Number of graph expansion rounds |
| `--skip-expansion` | `false` | Skip graph expansion phase |
| `--skip-lessons` | `false` | Skip lesson content generation |
| `--enable-blockchain` | `false` | Generate blockchain integration |
| `--max-commits` | `2000` | Max commits to fetch for analysis |
| `-v, --verbose` | `false` | Enable debug logging |

## Generated course structure

```
course-repo/
├── CLAUDE.md                 # Tutor instructions for Claude Code
├── README.md                 # Setup guide for learners
├── .gitignore
├── .learner/
│   └── profile.json          # Learner profile
├── knowledge/
│   ├── graph.json            # Knowledge graph (nodes + edges)
│   └── courses.json          # Courses with lesson content
├── blockchain/               # (only with --enable-blockchain)
│   └── config.json           # AIN topic map, depth map, provider URL
└── src/                      # Code snippets referenced by lessons
```

## How learners use it

```bash
cd course-repo
claude                        # Start Claude Code
```

Then just chat:
- "show the graph" — see the knowledge graph with progress
- "teach me self-attention" — get a lesson with code, analogies, and a quiz
- "next" — get a recommendation for what to learn next
- "friends" — see other explorers on the blockchain
- "frontier" — see community progress stats

## Architecture

| Module | Role |
|--------|------|
| `repo_analyzer.py` | Scans repo structure, imports, commits |
| `concept_extractor.py` | LLM extracts concepts → knowledge graph |
| `expander.py` | LLM proposes frontier concepts over multiple rounds |
| `course_builder.py` | LLM generates courses and lesson content |
| `scaffold.py` | Generates the learner-facing course repo |
| `graph.py` | Knowledge graph data structure and operations |
| `models.py` | Data models (Concept, Lesson, Course, LearnerProfile) |
| `llm_client.py` | vLLM API client |
| `cli.py` | CLI entry point and argument parsing |

## Blockchain integration

When `--enable-blockchain` is enabled, the generated course uses the [AIN blockchain](https://ainetwork.ai/) Knowledge module:

- **Progress tracking**: Concept completions are recorded on-chain via `ain.knowledge.explore()`
- **Explorer discovery**: Learners can see who else explored each topic via `ain.knowledge.getExplorers()`
- **Frontier map**: Community-wide stats per topic (explorer count, max depth, avg depth)
- **x402 payment gating**: Premium lessons can require on-chain payment via `ain.knowledge.access()`

The course repo contains only a `config.json` with topic mappings — no helper scripts. The tutor uses the local `ain-js` library directly via inline Node.js commands.

## License

MIT
