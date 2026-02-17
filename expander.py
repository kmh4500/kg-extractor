"""BFS expansion of the knowledge graph to frontier/cutting-edge concepts."""

from __future__ import annotations

import logging
from typing import Optional

from kg_extractor.graph import KnowledgeGraph
from kg_extractor.llm_client import get_client, chat_completion, parse_json_response
from kg_extractor.models import (
    ConceptNode, ConceptType, ConceptLevel, Edge, RelationshipType,
)

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = """\
You are a transformer architecture expert. Given a set of existing concepts in a \
knowledge graph, identify NEW cutting-edge concepts that extend from these concepts \
but are not yet in the graph.

Focus on:
- Latest model architectures (Mixtral, Qwen2, Gemma2, DeepSeek-V2, Command R, etc.)
- Cutting-edge techniques (ring attention, speculative decoding, MoE routing, etc.)
- Recent innovations in efficiency (quantization advances, KV cache optimization, etc.)
- Emerging paradigms (state space models, hybrid architectures, etc.)

For each new concept, provide:
- id: snake_case identifier
- name: human-readable name
- type: one of {types}
- level: one of {levels}
- description: 1-2 sentence description
- key_ideas: list of 2-4 key ideas
- confidence: float 0.0-1.0 (how certain you are this belongs in the graph)

For each new edge connecting new concepts to existing ones, provide:
- source: concept id
- target: concept id
- relationship: one of {relationships}
- description: brief description

Return ONLY valid JSON with keys "new_nodes" and "new_edges". Only include truly new \
concepts not already in the provided list. No other text.\
"""

EXPANSION_USER_PROMPT = """\
Here are the existing concepts in the knowledge graph:

{existing_concepts}

Identify {num_new} new cutting-edge concepts that extend from these, especially \
focusing on frontier models and techniques from 2024-2025. Include proper edges \
connecting them to existing concepts.

Return ONLY valid JSON with keys "new_nodes" and "new_edges".\
"""


class GraphExpander:
    """Expands the knowledge graph to include frontier concepts via BFS."""

    def __init__(self, base_url: Optional[str] = None, model: str = "google/gemma-3-27b-it"):
        self.client = get_client(base_url)
        self.model = model

    def expand(self, kg: KnowledgeGraph, rounds: int = 2, concepts_per_round: int = 10) -> KnowledgeGraph:
        """Expand the graph through multiple BFS rounds."""
        for round_num in range(1, rounds + 1):
            logger.info("Expansion round %d/%d", round_num, rounds)
            new_nodes, new_edges = self._expand_one_round(kg, concepts_per_round)

            if not new_nodes:
                logger.info("No new concepts found, stopping expansion")
                break

            for node in new_nodes:
                kg.add_concept(node)
            for edge in new_edges:
                kg.add_edge(edge)

            logger.info(
                "Round %d: added %d concepts, %d edges",
                round_num, len(new_nodes), len(new_edges),
            )

        return kg

    def _expand_one_round(self, kg: KnowledgeGraph, num_new: int) -> tuple[list[ConceptNode], list[Edge]]:
        """Run one round of expansion."""
        system_prompt = EXPANSION_SYSTEM_PROMPT.format(
            types=", ".join(t.value for t in ConceptType),
            levels=", ".join(l.value for l in ConceptLevel),
            relationships=", ".join(r.value for r in RelationshipType),
        )

        existing = "\n".join(
            f"- {n.id}: {n.name} ({n.type.value}, {n.level.value}) — {n.description[:100]}"
            for n in kg.get_all_concepts()
        )

        user_prompt = EXPANSION_USER_PROMPT.format(
            existing_concepts=existing,
            num_new=num_new,
        )

        response_text = chat_completion(
            self.client, self.model, system_prompt, user_prompt,
            max_tokens=4096, temperature=0.3,
        )

        data = parse_json_response(response_text)

        existing_ids = {n.id for n in kg.get_all_concepts()}
        new_nodes = []
        for nd in data.get("new_nodes", []):
            if nd.get("id") in existing_ids:
                continue
            try:
                node = ConceptNode(
                    id=nd["id"],
                    name=nd["name"],
                    type=ConceptType(nd.get("type", "architecture")),
                    level=ConceptLevel(nd.get("level", "frontier")),
                    description=nd.get("description", ""),
                    key_ideas=nd.get("key_ideas", []),
                    code_refs=nd.get("code_refs", []),
                    confidence=nd.get("confidence", 0.8),
                )
                new_nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid expansion node: %s", e)

        new_node_ids = {n.id for n in new_nodes}
        all_valid = existing_ids | new_node_ids

        new_edges = []
        for ed in data.get("new_edges", []):
            try:
                source = ed["source"]
                target = ed["target"]
                if source not in all_valid or target not in all_valid:
                    continue
                edge = Edge(
                    source=source,
                    target=target,
                    relationship=RelationshipType(ed.get("relationship", "builds_on")),
                    description=ed.get("description", ""),
                )
                new_edges.append(edge)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid expansion edge: %s", e)

        return new_nodes, new_edges
