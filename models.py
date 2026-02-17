"""Data models for the knowledge graph and course system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ConceptType(str, Enum):
    ARCHITECTURE = "architecture"
    TECHNIQUE = "technique"
    COMPONENT = "component"
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    TOKENIZATION = "tokenization"
    THEORY = "theory"
    APPLICATION = "application"


class RelationshipType(str, Enum):
    BUILDS_ON = "builds_on"
    OPTIMIZES = "optimizes"
    REQUIRES = "requires"
    EVOLVES_TO = "evolves_to"
    VARIANT_OF = "variant_of"
    COMPONENT_OF = "component_of"
    ALTERNATIVE_TO = "alternative_to"
    ENABLES = "enables"


class ConceptLevel(str, Enum):
    FOUNDATIONAL = "foundational"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FRONTIER = "frontier"


# Maps concept level to AIN exploration depth (1-4)
CONCEPT_LEVEL_DEPTH: dict[str, int] = {
    ConceptLevel.FOUNDATIONAL: 1,
    ConceptLevel.INTERMEDIATE: 2,
    ConceptLevel.ADVANCED: 3,
    ConceptLevel.FRONTIER: 4,
}


@dataclass
class ConceptNode:
    id: str
    name: str
    type: ConceptType
    level: ConceptLevel
    description: str
    key_ideas: list[str] = field(default_factory=list)
    code_refs: list[str] = field(default_factory=list)
    paper_ref: str = ""  # e.g. "Vaswani et al., 2017 — Attention Is All You Need"
    first_appeared: Optional[str] = None  # date string or paper year
    confidence: float = 1.0  # 1.0 for repo-sourced, lower for expanded

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d["level"] = self.level.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ConceptNode:
        d = d.copy()
        d["type"] = ConceptType(d["type"])
        d["level"] = ConceptLevel(d["level"])
        d.setdefault("paper_ref", "")
        return cls(**d)


@dataclass
class Edge:
    source: str  # concept id
    target: str  # concept id
    relationship: RelationshipType
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relationship"] = self.relationship.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        d = d.copy()
        d["relationship"] = RelationshipType(d["relationship"])
        return cls(**d)


@dataclass
class Lesson:
    concept_id: str
    title: str
    prerequisites: list[str] = field(default_factory=list)
    key_ideas: list[str] = field(default_factory=list)
    code_ref: str = ""
    paper_ref: str = ""
    exercise: str = ""
    explanation: str = ""
    x402_price: str = ""
    x402_gateway: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Lesson:
        d = d.copy()
        d.setdefault("paper_ref", "")
        d.setdefault("x402_price", "")
        d.setdefault("x402_gateway", "")
        return cls(**d)


@dataclass
class Course:
    id: str
    title: str
    description: str = ""
    concepts: list[str] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lessons"] = [lesson.to_dict() for lesson in self.lessons]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Course:
        d = d.copy()
        d["lessons"] = [Lesson.from_dict(l) for l in d.get("lessons", [])]
        return cls(**d)


@dataclass
class LearnerProfile:
    name: str = "Learner"
    avatar: str = "🧑‍💻"
    started_at: str = ""
    git_user: str = ""
    wallet_address: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = date.today().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LearnerProfile:
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> LearnerProfile:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class LearnerProgress:
    current_concept: str = ""
    completed: list[str] = field(default_factory=list)
    in_progress: list[str] = field(default_factory=list)
    started_at: str = ""
    last_active: str = ""

    def __post_init__(self):
        now = date.today().isoformat()
        if not self.started_at:
            self.started_at = now
        if not self.last_active:
            self.last_active = now

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LearnerProgress:
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> LearnerProgress:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class RepoAnalysis:
    """Results from mining the HF Transformers repo."""
    models: list[dict] = field(default_factory=list)       # model name, path, first commit date
    components: list[dict] = field(default_factory=list)    # shared components found
    key_commits: list[dict] = field(default_factory=list)   # important evolution commits
    doc_summaries: list[dict] = field(default_factory=list) # per-model doc summaries

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RepoAnalysis:
        return cls(**d)
