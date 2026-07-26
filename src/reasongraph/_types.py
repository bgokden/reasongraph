from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Node:
    """A node in the reason graph.

    ``scopes`` is a set of free-text tags (e.g. ``{"user-123", "topic-economy"}``).
    Scopes are labels, not partitions: the graph is shared and traversal crosses
    scopes freely, so a scope only narrows where a query starts, not what it can
    reach. A node may carry any number of scopes, or none.
    """

    content: str
    type: str = "text"
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    scopes: set[str] = field(default_factory=set)
    # When a fact is soft-superseded (retired) it stays in the graph but is stamped
    # with the time it stopped being current. ``None`` means still valid. Together
    # with ``created_at`` this gives each fact a validity interval, so a query can
    # exclude retired facts by default and time-travel to "what was current at T".
    invalid_at: datetime | None = None


@dataclass
class Edge:
    """A directed edge between two nodes.

    ``label`` types the edge. Most edges are untyped (``None``) structural links
    (entity->text, text->text). A ``"causes"`` label marks a directed
    cause->effect link, making causality first-class and distinguishable from an
    anonymous entity bridge.
    """

    from_content: str
    to_content: str
    last_accessed: datetime = field(default_factory=datetime.now)
    label: str | None = None
