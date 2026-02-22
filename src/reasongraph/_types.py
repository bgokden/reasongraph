from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Node:
    """A node in the reason graph."""

    content: str
    type: str = "text"
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class Edge:
    """A directed edge between two nodes."""

    from_content: str
    to_content: str
    last_accessed: datetime = field(default_factory=datetime.now)
