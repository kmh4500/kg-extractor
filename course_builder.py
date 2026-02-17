"""Build structured courses from the knowledge graph."""

from __future__ import annotations

import logging
from typing import Optional

from kg_extractor.graph import KnowledgeGraph
from kg_extractor.llm_client import get_client, chat_completion, parse_json_response
from kg_extractor.models import (
    ConceptLevel, ConceptNode, Course, Lesson, RelationshipType,
)

logger = logging.getLogger(__name__)

# Course cluster definitions
COURSE_CLUSTERS = [
    {
        "id": "foundations",
        "title": "Transformer Foundations",
        "description": "Core concepts every transformer practitioner must understand",
        "levels": [ConceptLevel.FOUNDATIONAL],
        "priority_types": ["theory", "component"],
    },
    {
        "id": "encoder_models",
        "title": "Encoder Models (BERT Family)",
        "description": "Understanding bidirectional transformers and their applications",
        "levels": [ConceptLevel.INTERMEDIATE],
        "keywords": ["bert", "roberta", "electra", "deberta", "albert", "distilbert", "encoder"],
    },
    {
        "id": "decoder_models",
        "title": "Decoder Models (GPT Family)",
        "description": "Autoregressive language models from GPT to modern LLMs",
        "levels": [ConceptLevel.INTERMEDIATE, ConceptLevel.ADVANCED],
        "keywords": ["gpt", "llama", "mistral", "falcon", "opt", "bloom", "decoder", "causal", "autoregressive"],
    },
    {
        "id": "seq2seq_models",
        "title": "Sequence-to-Sequence Models",
        "description": "Encoder-decoder architectures for translation and generation",
        "levels": [ConceptLevel.INTERMEDIATE],
        "keywords": ["t5", "bart", "marian", "pegasus", "seq2seq", "encoder_decoder"],
    },
    {
        "id": "efficiency",
        "title": "Efficiency & Optimization",
        "description": "Making transformers faster and smaller",
        "levels": [ConceptLevel.ADVANCED],
        "keywords": [
            "attention", "flash", "quantization", "pruning", "distillation",
            "efficient", "sparse", "linear", "cache", "optimization",
        ],
    },
    {
        "id": "frontier",
        "title": "Frontier Models & Techniques",
        "description": "Cutting-edge architectures and emerging paradigms",
        "levels": [ConceptLevel.FRONTIER],
    },
]

LESSON_GENERATION_PROMPT = """\
You are writing ONE short lesson for an interactive transformer course delivered \
inside Claude Code (a CLI chat — the learner cannot open files or run GUIs).

## Concept
Name: {concept_name}
Paper: {paper_ref}
Description: {concept_description}
Key ideas: {key_ideas}
Code references: {code_refs}
Prerequisites: {prerequisites}

## Rules for the "explanation" field (MUST be under 800 words)
1. **Paper-first**: Open with the paper/origin — who wrote it, what year, what \
problem it solved, and why it mattered. If no paper is known, open with the \
core problem the concept addresses.
2. **Short paragraphs**: 2-3 sentences max per paragraph. Use at most 3 paragraphs total.
3. **One vivid analogy**: Include exactly one concrete analogy or mental image \
(e.g. "Think of attention as a spotlight sweeping over words").
4. **Inline code**: Show small code snippets (< 10 lines) directly in the text \
using markdown fenced blocks. NEVER say "open the file" or "look at file X" — \
the learner cannot open files.
5. **No slash commands**: Never write /command — the learner talks to the tutor \
in plain English.
6. **Be concise**: The entire explanation MUST fit in under 800 words.

## Rules for the "exercise" field
Write ONE quiz-style exercise the learner can answer by typing a number or a short \
sentence. The learner is chatting — they should NOT have to write code.

GOOD formats (pick one):
- **Multiple choice**: "Which of these is true about X?\\n1. ...\\n2. ...\\n3. ...\\n4. ...\\nType the number."
- **Predict the output**: Show a small (< 8 line) code snippet and ask "What does this print?" The answer is a value, not code.
- **Fill in the blank**: "In the formula Attention(Q,K,V) = softmax(QK^T / ____)V, what goes in the blank and why?"
- **Short answer**: "In one sentence, why does X matter for Y?"
- **True or false**: "True or false: ..."

BAD (never use these):
- "Write a function …" / "Implement …" — too hard for a chat quiz
- "Explore the implementation of …" — too vague
- "Open src/… and read …" — impossible in chat
- "Run /exercise" — not a real command

Return ONLY valid JSON with keys "explanation" and "exercise". No other text.\
"""


class CourseBuilder:
    """Builds structured courses from the knowledge graph."""

    def __init__(self, base_url: Optional[str] = None, model: str = "google/gemma-3-27b-it"):
        self.client = get_client(base_url)
        self.model = model

    def build_courses(self, kg: KnowledgeGraph, generate_lessons: bool = True) -> list[Course]:
        """Build courses from the knowledge graph."""
        sorted_concepts = kg.topological_sort()
        courses = self._cluster_concepts(kg, sorted_concepts)

        if generate_lessons:
            for course in courses:
                logger.info("Generating lessons for course: %s", course.title)
                course.lessons = self._generate_lessons(kg, course.concepts)

        courses = [c for c in courses if c.concepts]

        logger.info("Built %d courses with %d total concepts",
                     len(courses),
                     sum(len(c.concepts) for c in courses))
        return courses

    def _cluster_concepts(self, kg: KnowledgeGraph, sorted_ids: list[str]) -> list[Course]:
        assigned = set()
        courses = []

        for cluster in COURSE_CLUSTERS:
            course = Course(
                id=cluster["id"],
                title=cluster["title"],
                description=cluster.get("description", ""),
            )

            for concept_id in sorted_ids:
                if concept_id in assigned:
                    continue
                node = kg.get_concept(concept_id)
                if not node:
                    continue

                if self._concept_matches_cluster(node, cluster):
                    course.concepts.append(concept_id)
                    assigned.add(concept_id)

            courses.append(course)

        for concept_id in sorted_ids:
            if concept_id not in assigned:
                node = kg.get_concept(concept_id)
                if node:
                    best_course = self._find_best_course(node, courses)
                    best_course.concepts.append(concept_id)

        return courses

    def _concept_matches_cluster(self, node: ConceptNode, cluster: dict) -> bool:
        levels = cluster.get("levels", [])
        if levels and node.level not in levels:
            return False

        keywords = cluster.get("keywords", [])
        if keywords:
            node_text = f"{node.id} {node.name} {node.description}".lower()
            return any(kw in node_text for kw in keywords)

        priority_types = cluster.get("priority_types", [])
        if priority_types:
            return node.type.value in priority_types

        return bool(levels)

    def _find_best_course(self, node: ConceptNode, courses: list[Course]) -> Course:
        level_to_course = {
            ConceptLevel.FOUNDATIONAL: "foundations",
            ConceptLevel.INTERMEDIATE: "encoder_models",
            ConceptLevel.ADVANCED: "efficiency",
            ConceptLevel.FRONTIER: "frontier",
        }
        target_id = level_to_course.get(node.level, "frontier")
        for course in courses:
            if course.id == target_id:
                return course
        return courses[-1]

    def _generate_lessons(self, kg: KnowledgeGraph, concept_ids: list[str]) -> list[Lesson]:
        lessons = []
        for concept_id in concept_ids:
            node = kg.get_concept(concept_id)
            if not node:
                continue

            prereqs = kg.get_prerequisites(concept_id)
            prereq_names = []
            for pid in prereqs:
                pnode = kg.get_concept(pid)
                if pnode:
                    prereq_names.append(pnode.name)

            lesson = self._generate_one_lesson(node, prereq_names)
            lessons.append(lesson)

        return lessons

    def _generate_one_lesson(self, node: ConceptNode, prerequisite_names: list[str]) -> Lesson:
        fallback_exercise = (
            f"True or false: {node.name} was introduced to solve a problem with "
            "earlier approaches. Explain your answer in one sentence."
        )

        prompt = LESSON_GENERATION_PROMPT.format(
            concept_name=node.name,
            paper_ref=node.paper_ref or "unknown",
            concept_description=node.description,
            key_ideas=", ".join(node.key_ideas) if node.key_ideas else "N/A",
            code_refs=", ".join(node.code_refs) if node.code_refs else "N/A",
            prerequisites=", ".join(prerequisite_names) if prerequisite_names else "None",
        )

        try:
            text = chat_completion(
                self.client, self.model, "", prompt,
                max_tokens=6144, temperature=0.3,
            )
            data = parse_json_response(text)

            return Lesson(
                concept_id=node.id,
                title=node.name,
                prerequisites=[],
                key_ideas=node.key_ideas,
                code_ref=node.code_refs[0] if node.code_refs else "",
                paper_ref=node.paper_ref,
                exercise=data.get("exercise", fallback_exercise),
                explanation=data.get("explanation", node.description),
            )
        except Exception as e:
            logger.warning("Failed to generate lesson for %s: %s", node.id, e)
            return Lesson(
                concept_id=node.id,
                title=node.name,
                prerequisites=[],
                key_ideas=node.key_ideas,
                code_ref=node.code_refs[0] if node.code_refs else "",
                paper_ref=node.paper_ref,
                exercise=fallback_exercise,
                explanation=node.description,
            )
