"""NetworkX-based knowledge graph for transformer concepts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx

from kg_extractor.models import ConceptNode, Edge, ConceptLevel, RelationshipType


class KnowledgeGraph:
    """Manages the concept knowledge graph using NetworkX."""

    def __init__(self):
        self.g = nx.DiGraph()
        self._nodes: dict[str, ConceptNode] = {}
        self._edges: list[Edge] = []

    def add_concept(self, node: ConceptNode) -> None:
        self._nodes[node.id] = node
        self.g.add_node(node.id, **node.to_dict())

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)
        self.g.add_edge(
            edge.source,
            edge.target,
            relationship=edge.relationship.value,
            weight=edge.weight,
            description=edge.description,
        )

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        return self._nodes.get(concept_id)

    def get_prerequisites(self, concept_id: str) -> list[str]:
        """Get all concepts that this concept requires (predecessors via REQUIRES/BUILDS_ON)."""
        prereqs = []
        for pred in self.g.predecessors(concept_id):
            edge_data = self.g.edges[pred, concept_id]
            rel = edge_data.get("relationship", "")
            if rel in (RelationshipType.REQUIRES.value, RelationshipType.BUILDS_ON.value):
                prereqs.append(pred)
        return prereqs

    def get_dependents(self, concept_id: str) -> list[str]:
        """Get concepts that depend on this concept."""
        deps = []
        for succ in self.g.successors(concept_id):
            edge_data = self.g.edges[concept_id, succ]
            rel = edge_data.get("relationship", "")
            if rel in (RelationshipType.REQUIRES.value, RelationshipType.BUILDS_ON.value):
                deps.append(succ)
        return deps

    def topological_sort(self) -> list[str]:
        """Return concepts in topological order (prerequisites first)."""
        try:
            return list(nx.topological_sort(self.g))
        except nx.NetworkXUnfeasible:
            # Graph has cycles — fall back to level-based ordering
            return self._level_based_sort()

    def _level_based_sort(self) -> list[str]:
        level_order = {
            ConceptLevel.FOUNDATIONAL.value: 0,
            ConceptLevel.INTERMEDIATE.value: 1,
            ConceptLevel.ADVANCED.value: 2,
            ConceptLevel.FRONTIER.value: 3,
        }
        nodes = list(self._nodes.values())
        nodes.sort(key=lambda n: (level_order.get(n.level.value, 99), n.id))
        return [n.id for n in nodes]

    def get_concepts_by_level(self, level: ConceptLevel) -> list[ConceptNode]:
        return [n for n in self._nodes.values() if n.level == level]

    def get_all_concepts(self) -> list[ConceptNode]:
        return list(self._nodes.values())

    def get_all_edges(self) -> list[Edge]:
        return list(self._edges)

    def get_frontier_concepts(self) -> list[ConceptNode]:
        """Get leaf concepts (no outgoing prerequisite edges)."""
        frontier = []
        for node_id in self.g.nodes():
            successors = list(self.g.successors(node_id))
            if not successors:
                node = self._nodes.get(node_id)
                if node:
                    frontier.append(node)
        return frontier

    def get_root_concepts(self) -> list[ConceptNode]:
        """Get root concepts (no incoming prerequisite edges)."""
        roots = []
        for node_id in self.g.nodes():
            predecessors = list(self.g.predecessors(node_id))
            if not predecessors:
                node = self._nodes.get(node_id)
                if node:
                    roots.append(node)
        return roots

    def subgraph(self, concept_ids: list[str]) -> KnowledgeGraph:
        """Create a subgraph containing only the specified concepts."""
        sub = KnowledgeGraph()
        for cid in concept_ids:
            node = self._nodes.get(cid)
            if node:
                sub.add_concept(node)
        for edge in self._edges:
            if edge.source in concept_ids and edge.target in concept_ids:
                sub.add_edge(edge)
        return sub

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeGraph:
        kg = cls()
        for node_data in d.get("nodes", []):
            kg.add_concept(ConceptNode.from_dict(node_data))
        for edge_data in d.get("edges", []):
            kg.add_edge(Edge.from_dict(edge_data))
        return kg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        return cls.from_dict(json.loads(path.read_text()))

    def to_mermaid(self, completed: list[str] = None, current: str = None,
                   friends: dict[str, dict] = None) -> str:
        """Render graph as Mermaid diagram with progress markers.

        Args:
            completed: list of completed concept IDs
            current: current concept ID
            friends: dict of {friend_name: {"avatar": "🦊", "current_concept": "..."}}
        """
        completed = completed or []
        friends = friends or {}

        lines = ["graph LR"]
        # Build friend positions: concept_id -> list of avatars
        friend_positions: dict[str, list[str]] = {}
        for fname, fdata in friends.items():
            fc = fdata.get("current_concept", "")
            if fc:
                friend_positions.setdefault(fc, []).append(fdata.get("avatar", "👤"))

        for node in self._nodes.values():
            label = node.name
            markers = []
            if node.id in completed:
                markers.append("✅")
            if node.id == current:
                markers.append("👈")
            avatars = friend_positions.get(node.id, [])
            if avatars:
                markers.extend(avatars)
            if markers:
                label += " " + "".join(markers)
            lines.append(f"    {node.id}[{label}]")

        for edge in self._edges:
            if edge.source in self._nodes and edge.target in self._nodes:
                lines.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "num_concepts": len(self._nodes),
            "num_edges": len(self._edges),
            "num_foundational": len(self.get_concepts_by_level(ConceptLevel.FOUNDATIONAL)),
            "num_intermediate": len(self.get_concepts_by_level(ConceptLevel.INTERMEDIATE)),
            "num_advanced": len(self.get_concepts_by_level(ConceptLevel.ADVANCED)),
            "num_frontier": len(self.get_concepts_by_level(ConceptLevel.FRONTIER)),
        }
