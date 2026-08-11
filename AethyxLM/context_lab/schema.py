"""Representation-neutral schemas for the local context compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ContextKind = Literal[
    "system",
    "tool_schema",
    "conversation",
    "tool_result",
    "document",
    "agent_state",
    "metadata",
]


@dataclass(frozen=True)
class ContextItem:
    source_id: str
    kind: ContextKind
    text: str
    priority: float = 0.5
    protected: bool = False
    recency: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_id or not self.text:
            raise ValueError("context items require source_id and text")
        if not 0 <= self.priority <= 1:
            raise ValueError("priority must be within [0, 1]")


@dataclass(frozen=True)
class ContextRequest:
    query: str
    items: tuple[ContextItem, ...]
    request_id: str = ""

    def __post_init__(self):
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if not self.items:
            raise ValueError("at least one context item is required")
        ids = [item.source_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    kind: str
    source_ids: tuple[str, ...]
    importance: float


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    provenance: Literal["EXTRACTED", "INFERRED"]
    confidence: float
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class VisualPagePlan:
    page_id: str
    source_ids: tuple[str, ...]
    text: str
    width: int
    height: int
    columns: int
    estimated_image_units: int


@dataclass(frozen=True)
class CompiledContext:
    version: str
    mode: str
    query: str
    protected_items: tuple[ContextItem, ...]
    selected_items: tuple[ContextItem, ...]
    graph_nodes: tuple[GraphNode, ...]
    graph_edges: tuple[GraphEdge, ...]
    visual_pages: tuple[VisualPagePlan, ...]
    omitted_source_ids: tuple[str, ...]
    source_digest: str
    warnings: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        """Serializable research envelope; visual text is excluded from transport."""
        return {
            "version": self.version,
            "mode": self.mode,
            "query": self.query,
            "protected_items": [asdict(item) for item in self.protected_items],
            "selected_items": [asdict(item) for item in self.selected_items],
            "graph": {
                "nodes": [asdict(node) for node in self.graph_nodes],
                "edges": [asdict(edge) for edge in self.graph_edges],
            },
            "visual_pages": [
                {
                    "page_id": page.page_id,
                    "source_ids": page.source_ids,
                    "width": page.width,
                    "height": page.height,
                    "columns": page.columns,
                    "estimated_image_units": page.estimated_image_units,
                }
                for page in self.visual_pages
            ],
            "omitted_source_ids": self.omitted_source_ids,
            "source_digest": self.source_digest,
            "warnings": self.warnings,
        }

    def model_payload(self) -> dict[str, Any]:
        """Compact payload intended for representation-cost measurement."""
        if self.mode == "raw":
            return {
                "v": "r1",
                "q": self.query,
                "t": "\n".join(item.text for item in self.protected_items),
            }
        protected = [[item.kind, item.text] for item in self.protected_items]
        node_index = {
            node.node_id: index for index, node in enumerate(self.graph_nodes)
        }
        nodes = [
            [node.kind, node.label]
            for node in self.graph_nodes
        ]
        edges = [
            [
                node_index[edge.source],
                node_index[edge.target],
                edge.relation,
                "E" if edge.provenance == "EXTRACTED" else "I",
            ]
            for edge in self.graph_edges
            if edge.source in node_index and edge.target in node_index
        ]
        pages = [
            [page.page_id, page.width, page.height]
            for page in self.visual_pages
        ]
        return {
            "v": "ac1",
            "q": self.query,
            "p": protected,
            "n": nodes,
            "e": edges,
            "i": pages,
        }
