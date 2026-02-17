"""Generate the course repo scaffold that students clone and use with Claude Code."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from kg_extractor.graph import KnowledgeGraph
from kg_extractor.models import (
    CONCEPT_LEVEL_DEPTH,
    Course,
    LearnerProfile,
    LearnerProgress,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Templates — base (non-blockchain)
# ---------------------------------------------------------------------------

CLAUDE_MD_TEMPLATE = """\
# Transformer Learning Path

You are a friendly, knowledgeable tutor for this course.

## Data files
- Knowledge graph: knowledge/graph.json
- Courses & lessons: knowledge/courses.json
- Learner progress: .learner/progress.json
- Learner profile: .learner/profile.json

## How the learner talks to you
The learner just chats — no slash commands. Recognise these intents:
- "explore" / "show the graph" — render the knowledge graph as a Mermaid diagram,
  marking completed concepts with a checkmark and current concept with an arrow.
- "status" — show profile, completion %, current concept, and friends' positions.
- "learn <concept>" or "teach me <concept>" — deliver the lesson (see teaching
  style below).
- "exercise" / "give me a challenge" — present the exercise for the current concept.
- "done" / "I finished" — mark the current concept as completed, suggest next.
- "friends" — list friends and their progress.
- "next" / "what should I learn next?" — recommend the next concept via
  prerequisites and graph topology.
- "graph" — show full Mermaid graph of the current course.

## Teaching style (important!)
When teaching a concept:
1. **Paper-first**: Start with the paper or origin — who wrote it, when, and what
   problem it solved. If a lesson has a paper_ref field, cite it.
2. **Short paragraphs**: 2-3 sentences max. Dense walls of text lose people.
3. **Inline code**: Show small code snippets (< 15 lines) directly in your
   message using fenced code blocks. NEVER say "open the file" or "look at
   file X" — the learner is in a CLI chat and cannot open files.
4. **One vivid analogy**: Include one concrete analogy or mental image to make
   the concept stick.
5. **Quiz exercise**: End with a quiz the learner can answer by typing a number
   or a short sentence — multiple choice, predict-the-output, fill-in-the-blank,
   or true/false. Never ask the learner to write code (too hard for a chat).
   Never say "Explore the implementation of …" — that is too vague.
6. **Fun**: Be encouraging, use light humour, celebrate progress.

## Progress tracking
- Read .learner/progress.json for current state.
- Update it when learner completes concepts.
- Read .learner/profile.json for learner identity.

## Friends
- Friends share their .learner/ data via git branches or a shared remote.
- Check branches matching pattern "learner/*" for friends' progress files.
- Show their emoji avatar + current concept when requested.

## Graph structure
- Nodes have: id, name, type, level, description, key_ideas, code_refs, paper_ref
- Edges have: source, target, relationship (builds_on, requires, optimizes, etc.)
- Levels: foundational -> intermediate -> advanced -> frontier
"""

CLAUDE_MD_BLOCKCHAIN_TEMPLATE = """\
# Transformer Learning Path

You are a friendly, knowledgeable tutor for this course.

## Data files (read-only reference)
- Knowledge graph: knowledge/graph.json
- Courses & lessons: knowledge/courses.json
- Learner profile: .learner/profile.json
- Blockchain config: blockchain/config.json (provider_url, topic_map, depth_map)

## Progress tracking — blockchain is the source of truth
NEVER write to .learner/progress.json or any JSON file to track progress.
All progress is recorded on the AIN blockchain using ain-js directly.

Read blockchain/config.json for:
- `ain_js_path`: path to local ain-js library
- `provider_url`: AIN node URL
- `topic_map`: concept_id → AIN topic path
- `depth_map`: concept_id → exploration depth (1-4)

### ain-js API (use via inline node -e scripts)

All commands follow this pattern — load config, init Ain, load wallet, call API:
```bash
node -e "
  const Ain = require(require('./blockchain/config.json').ain_js_path).default;
  const cfg = require('./blockchain/config.json');
  const ain = new Ain(cfg.provider_url);
  const fs = require('fs');
  const pk = fs.readFileSync('blockchain/.env','utf-8').match(/AIN_PRIVATE_KEY=(.+)/)[1].trim();
  ain.wallet.addAndSetDefaultAccount(pk);
  // ... then call ain.knowledge methods
"
```

Key ain.knowledge methods:
- `ain.knowledge.explore(input)` — record an exploration on-chain
  - input: `{topicPath, title, content, summary, depth, tags}`
- `ain.knowledge.getExplorers(topicPath)` — list addresses that explored a topic
- `ain.knowledge.getExplorations(address, topicPath)` — get explorations by user
- `ain.knowledge.getFrontierMap(topicPrefix)` — per-topic stats (explorer_count, max_depth, avg_depth)
- `ain.knowledge.getTopicStats(topicPath)` — stats for one topic
- `ain.knowledge.access(ownerAddr, topicPath, entryId)` — access gated content (x402)

### Setup wallet (first time)
```bash
node -e "
  const Ain = require(require('./blockchain/config.json').ain_js_path).default;
  const cfg = require('./blockchain/config.json');
  const ain = new Ain(cfg.provider_url);
  const crypto = require('crypto'), fs = require('fs');
  let pk;
  try { pk = fs.readFileSync('blockchain/.env','utf-8').match(/AIN_PRIVATE_KEY=(.+)/)[1].trim(); }
  catch(e) { pk = crypto.randomBytes(32).toString('hex'); fs.writeFileSync('blockchain/.env','AIN_PRIVATE_KEY='+pk+'\\n'); }
  const addr = ain.wallet.addAndSetDefaultAccount(pk);
  const profile = JSON.parse(fs.readFileSync('.learner/profile.json','utf-8'));
  profile.wallet_address = addr;
  fs.writeFileSync('.learner/profile.json', JSON.stringify(profile,null,2)+'\\n');
  console.log(JSON.stringify({address: addr, status: 'ready'}));
"
```

### Record concept completion
Look up the concept's topicPath and depth from blockchain/config.json, then:
```bash
node -e "
  const Ain = require(require('./blockchain/config.json').ain_js_path).default;
  const cfg = require('./blockchain/config.json');
  const ain = new Ain(cfg.provider_url);
  const fs = require('fs');
  const pk = fs.readFileSync('blockchain/.env','utf-8').match(/AIN_PRIVATE_KEY=(.+)/)[1].trim();
  ain.wallet.addAndSetDefaultAccount(pk);
  ain.knowledge.explore({
    topicPath: cfg.topic_map['CONCEPT_ID'],
    title: 'TITLE',
    content: 'CONTENT',
    summary: 'SUMMARY',
    depth: cfg.depth_map['CONCEPT_ID'] || 1,
    tags: 'CONCEPT_ID'
  }).then(r => console.log(JSON.stringify(r)));
"
```
Replace CONCEPT_ID, TITLE, CONTENT, SUMMARY with actual values.

### Get frontier map
```bash
node -e "
  const Ain = require(require('./blockchain/config.json').ain_js_path).default;
  const cfg = require('./blockchain/config.json');
  const ain = new Ain(cfg.provider_url);
  ain.knowledge.getFrontierMap(cfg.topic_prefix).then(r => console.log(JSON.stringify(r, null, 2)));
"
```

### Get explorers for a concept
```bash
node -e "
  const Ain = require(require('./blockchain/config.json').ain_js_path).default;
  const cfg = require('./blockchain/config.json');
  const ain = new Ain(cfg.provider_url);
  ain.knowledge.getExplorers(cfg.topic_map['CONCEPT_ID']).then(r => console.log(JSON.stringify(r)));
"
```

## How the learner talks to you
The learner just chats — no slash commands. Recognise these intents:
- "explore" / "show the graph" — render the knowledge graph as a Mermaid diagram,
  marking completed concepts with a checkmark and current concept with an arrow.
  Use the frontier-map API to determine which are completed.
- "status" — show profile, completion stats from frontier-map, and explorers.
- "learn <concept>" or "teach me <concept>" — deliver the lesson (see teaching
  style below).
- "exercise" / "give me a challenge" — present the exercise for the current concept.
- "done" / "I finished" — record on-chain (see "Record concept completion" above).
- "friends" / "explorers" — show on-chain explorers via getExplorers API.
- "next" / "what should I learn next?" — recommend the next concept via
  prerequisites, graph topology, and what's already explored on-chain.
- "graph" — show full Mermaid graph of the current course.
- "frontier" — show on-chain stats per topic via getFrontierMap API.
- "setup wallet" — run wallet setup (see above).

## Teaching style (important!)
When teaching a concept:
1. **Paper-first**: Start with the paper or origin — who wrote it, when, and what
   problem it solved. If a lesson has a paper_ref field, cite it.
2. **Short paragraphs**: 2-3 sentences max. Dense walls of text lose people.
3. **Inline code**: Show small code snippets (< 15 lines) directly in your
   message using fenced code blocks. NEVER say "open the file" or "look at
   file X" — the learner is in a CLI chat and cannot open files.
4. **One vivid analogy**: Include one concrete analogy or mental image to make
   the concept stick.
5. **Quiz exercise**: End with a quiz the learner can answer by typing a number
   or a short sentence — multiple choice, predict-the-output, fill-in-the-blank,
   or true/false. Never ask the learner to write code (too hard for a chat).
   Never say "Explore the implementation of …" — that is too vague.
6. **Fun**: Be encouraging, use light humour, celebrate progress.

## Completing a concept
When the learner says "done" or finishes a quiz correctly:
1. Run the "Record concept completion" script above with the concept details.
2. Confirm to the learner that progress is recorded on-chain.
3. Use the knowledge graph to recommend the next concept based on prerequisites.

## Friends / Explorers (blockchain-powered)
Instead of git branches, friends are discovered on-chain:
- Use `ain.knowledge.getExplorers(topicPath)` to list wallet addresses.
- Use `ain.knowledge.getExplorations(address, topicPath)` to see what they wrote.
- Show addresses (or names if known) and their exploration summaries.

## Premium lessons (x402)
Some lessons have `x402_price` and `x402_gateway` fields in courses.json.
When the learner reaches a premium lesson:
1. Tell them the price and ask if they want to proceed.
2. If yes, use `ain.knowledge.access(ownerAddr, topicPath, entryId)`.

## Graph structure
- Nodes have: id, name, type, level, description, key_ideas, code_refs, paper_ref
- Edges have: source, target, relationship (builds_on, requires, optimizes, etc.)
- Levels: foundational -> intermediate -> advanced -> frontier
"""

GITIGNORE_TEMPLATE = """\
# Python
__pycache__/
*.pyc
*.pyo

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"""

GITIGNORE_BLOCKCHAIN_EXTRA = """\

# Blockchain
blockchain/node_modules/
blockchain/.env
"""

README_TEMPLATE = """\
# {title}

A Claude Code-powered interactive learning path for transformer architectures.

## Getting Started

1. Clone this repo
2. Open Claude Code in this directory:
   ```
   cd {dirname}/
   claude
   ```
3. Start learning — just chat naturally:
   ```
   explore              # see the knowledge graph
   teach me attention   # start your first lesson
   give me a challenge  # get a hands-on exercise
   done                 # mark complete, move on
   ```

## Sharing Progress with Friends

1. Create your learner branch:
   ```
   git checkout -b learner/your-name
   ```
2. As you learn, commit your progress:
   ```
   git add .learner/
   git commit -m "Progress update"
   git push origin learner/your-name
   ```
3. Fetch friends' branches to see their progress:
   ```
   git fetch --all
   friends
   ```

## Course Structure

{course_list}

## Stats

- {num_concepts} concepts across {num_courses} courses
- {num_foundational} foundational, {num_intermediate} intermediate, {num_advanced} advanced, {num_frontier} frontier concepts
"""

README_BLOCKCHAIN_TEMPLATE = """\
# {title}

A Claude Code-powered interactive learning path for transformer architectures,
with on-chain progress tracking via the AIN blockchain.

## Requirements

- [Claude Code](https://claude.com/claude-code)
- Node.js >= 16 (for blockchain helper)

## Getting Started

1. Clone this repo
2. Open Claude Code in this directory:
   ```
   cd {dirname}/
   claude
   ```
3. Set up your blockchain wallet (first time only):
   ```
   setup wallet
   ```
4. Start learning — just chat naturally:
   ```
   explore              # see the knowledge graph
   teach me attention   # start your first lesson
   give me a challenge  # get a hands-on exercise
   done                 # mark complete + record on-chain
   ```

## On-Chain Features

Your learning progress is recorded on the AIN blockchain. This enables:

- **Global discovery**: See who else is exploring the same topics.
- **Frontier map**: View community-wide stats per topic.
- **Premium content**: Some advanced lessons use x402 payment gating.

### Setup

```bash
cd blockchain
npm install
node ain-helper.js setup    # creates wallet, outputs your address
```

### Explorers (Friends)

Instead of git branches, friends are discovered on-chain:
```
explorers               # who else is learning this concept?
frontier                # community stats per topic
```

## Course Structure

{course_list}

## Stats

- {num_concepts} concepts across {num_courses} courses
- {num_foundational} foundational, {num_intermediate} intermediate, {num_advanced} advanced, {num_frontier} frontier concepts
"""


# ---------------------------------------------------------------------------
# Scaffolder
# ---------------------------------------------------------------------------


class Scaffolder:
    """Generates the course repo that students clone and use with Claude Code."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        courses: list[Course],
        enable_blockchain: bool = False,
    ):
        self.kg = kg
        self.courses = courses
        self.enable_blockchain = enable_blockchain

    def scaffold(self, output_dir: str | Path, repo_path: Optional[str | Path] = None) -> Path:
        """Generate the complete course repo.

        Args:
            output_dir: where to create the course repo
            repo_path: optional path to the source HF transformers repo for copying code snippets
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        logger.info("Scaffolding course repo at %s", output)

        self._write_claude_md(output)
        self._write_knowledge(output)
        self._write_learner_template(output)
        self._write_gitignore(output)
        self._write_readme(output)

        if self.enable_blockchain:
            self._write_blockchain(output)

        if repo_path:
            self._copy_code_snippets(output, Path(repo_path))

        logger.info("Course repo scaffolded at %s", output)
        return output

    def _write_claude_md(self, output: Path) -> None:
        template = CLAUDE_MD_BLOCKCHAIN_TEMPLATE if self.enable_blockchain else CLAUDE_MD_TEMPLATE
        (output / "CLAUDE.md").write_text(template)
        logger.info("Wrote CLAUDE.md")

    def _write_knowledge(self, output: Path) -> None:
        knowledge_dir = output / "knowledge"
        knowledge_dir.mkdir(exist_ok=True)

        # Write graph
        self.kg.save(knowledge_dir / "graph.json")
        logger.info("Wrote knowledge/graph.json")

        # Write courses
        courses_data = [c.to_dict() for c in self.courses]
        (knowledge_dir / "courses.json").write_text(
            json.dumps(courses_data, indent=2, ensure_ascii=False) + "\n"
        )
        logger.info("Wrote knowledge/courses.json")

    def _write_learner_template(self, output: Path) -> None:
        learner_dir = output / ".learner"
        learner_dir.mkdir(exist_ok=True)

        # Template profile
        profile = LearnerProfile(
            name="Your Name",
            avatar="\U0001f9d1\u200d\U0001f4bb",  # 🧑‍💻
            started_at=date.today().isoformat(),
            git_user="auto-detected",
        )
        profile.save(learner_dir / "profile.json")

        if not self.enable_blockchain:
            # Only write progress.json when NOT using blockchain
            # (blockchain mode uses on-chain state as source of truth)
            sorted_concepts = self.kg.topological_sort()
            first_concept = sorted_concepts[0] if sorted_concepts else ""

            progress = LearnerProgress(
                current_concept=first_concept,
                completed=[],
                in_progress=[],
                started_at=date.today().isoformat(),
                last_active=date.today().isoformat(),
            )
            progress.save(learner_dir / "progress.json")

        logger.info("Wrote .learner/ template")

    def _write_gitignore(self, output: Path) -> None:
        content = GITIGNORE_TEMPLATE
        if self.enable_blockchain:
            content += GITIGNORE_BLOCKCHAIN_EXTRA
        (output / ".gitignore").write_text(content)

    def _write_readme(self, output: Path) -> None:
        stats = self.kg.stats()
        course_list = "\n".join(
            f"- **{c.title}** ({len(c.concepts)} concepts): {c.description}"
            for c in self.courses if c.concepts
        )

        template = README_BLOCKCHAIN_TEMPLATE if self.enable_blockchain else README_TEMPLATE
        readme = template.format(
            title="Transformer Learning Path",
            dirname=output.name,
            course_list=course_list or "No courses generated yet.",
            num_concepts=stats["num_concepts"],
            num_courses=len([c for c in self.courses if c.concepts]),
            num_foundational=stats["num_foundational"],
            num_intermediate=stats["num_intermediate"],
            num_advanced=stats["num_advanced"],
            num_frontier=stats["num_frontier"],
        )
        (output / "README.md").write_text(readme)
        logger.info("Wrote README.md")

    # -----------------------------------------------------------------------
    # Blockchain scaffold
    # -----------------------------------------------------------------------

    def _write_blockchain(self, output: Path) -> None:
        """Create blockchain/ directory with config.json (topic mappings + ain-js path)."""
        bc_dir = output / "blockchain"
        bc_dir.mkdir(exist_ok=True)

        # Resolve the local ain-js lib path
        ain_js_lib = Path(__file__).resolve().parent.parent / "ain-js" / "lib" / "ain.js"
        if not ain_js_lib.exists():
            for candidate in [
                Path.home() / "git" / "ain-js" / "lib" / "ain.js",
                Path("/home/comcom/git/ain-js/lib/ain.js"),
            ]:
                if candidate.exists():
                    ain_js_lib = candidate
                    break

        # config.json — includes ain_js_path so the tutor can require() it
        config = self._build_blockchain_config()
        config["ain_js_path"] = str(ain_js_lib)
        (bc_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )

        logger.info("Wrote blockchain/config.json")

    def _build_blockchain_config(self) -> dict:
        """Build the blockchain config.json with topic map, depth map, etc."""
        # Derive a course-level prefix from the first course id, or default
        course_id = self.courses[0].id if self.courses else "transformers"
        topic_prefix = f"transformers/{course_id}"

        # Build topic_map: concept_id → full topic path
        topic_map: dict[str, str] = {}
        depth_map: dict[str, int] = {}
        topics_to_register: list[dict] = []

        # Register the root topic
        topics_to_register.append({
            "path": "transformers",
            "title": "Transformers",
            "description": "Transformer architecture learning topics",
        })
        topics_to_register.append({
            "path": topic_prefix,
            "title": course_id.replace("_", " ").title(),
            "description": f"Topics for {course_id} course",
        })

        seen_topics: set[str] = set()

        for node in self.kg.get_all_concepts():
            topic_path = f"{topic_prefix}/{node.id}"
            topic_map[node.id] = topic_path

            # Depth from level
            depth_map[node.id] = CONCEPT_LEVEL_DEPTH.get(node.level, 1)

            if topic_path not in seen_topics:
                seen_topics.add(topic_path)
                topics_to_register.append({
                    "path": topic_path,
                    "title": node.name,
                    "description": node.description[:200] if node.description else "",
                })

        # x402 lessons index
        x402_lessons: dict[str, dict] = {}
        for course in self.courses:
            for lesson in course.lessons:
                if lesson.x402_price:
                    x402_lessons[lesson.concept_id] = {
                        "price": lesson.x402_price,
                        "gateway": lesson.x402_gateway,
                    }

        return {
            "provider_url": "http://localhost:8081",
            "chain_id": 0,
            "topic_prefix": topic_prefix,
            "topic_map": topic_map,
            "depth_map": depth_map,
            "topics_to_register": topics_to_register,
            "x402_lessons": x402_lessons,
        }

    # -----------------------------------------------------------------------
    # Code snippets
    # -----------------------------------------------------------------------

    def _copy_code_snippets(self, output: Path, repo_path: Path) -> None:
        """Copy relevant code snippets from the source repo."""
        src_dir = output / "src"
        src_dir.mkdir(exist_ok=True)

        # Collect all code_refs from the graph
        code_refs = set()
        for node in self.kg.get_all_concepts():
            for ref in node.code_refs:
                # Parse "src/transformers/models/bert/modeling_bert.py:BertSelfAttention"
                file_part = ref.split(":")[0] if ":" in ref else ref
                code_refs.add(file_part)

        copied = 0
        for ref in code_refs:
            source_file = repo_path / ref
            if source_file.exists() and source_file.is_file():
                dest = src_dir / Path(ref).name
                try:
                    shutil.copy2(source_file, dest)
                    copied += 1
                except Exception as e:
                    logger.debug("Could not copy %s: %s", ref, e)

        logger.info("Copied %d code snippets to src/", copied)
