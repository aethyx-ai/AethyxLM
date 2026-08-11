"""Deterministic provenance-carrying graph extraction and scoped retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from context_lab.schema import ContextItem, GraphEdge, GraphNode


WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)
ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b")


def terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in WORD.finditer(text)}


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


class DeterministicContextGraph:
    """Small graph designed for context selection, not code intelligence."""

    def __init__(self, items: tuple[ContextItem, ...]):
        self.items = items
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self._build()

    def _build(self):
        entity_sources = defaultdict(list)
        previous = None
        for item in self.items:
            item_node = f"source:{item.source_id}"
            self.nodes.append(
                GraphNode(
                    item_node,
                    item.text[:160].replace("\n", " "),
                    item.kind,
                    (item.source_id,),
                    item.priority,
                )
            )
            if previous is not None:
                self.edges.append(
                    GraphEdge(
                        previous,
                        item_node,
                        "followed_by",
                        "EXTRACTED",
                        1.0,
                        (item.source_id,),
                    )
                )
            previous = item_node
            for entity in dict.fromkeys(ENTITY.findall(item.text)):
                entity_sources[entity].append(item.source_id)
                entity_node = stable_id("entity", entity.casefold())
                if not any(node.node_id == entity_node for node in self.nodes):
                    self.nodes.append(
                        GraphNode(entity_node, entity, "entity", (item.source_id,), 0.5)
                    )
                self.edges.append(
                    GraphEdge(
                        item_node,
                        entity_node,
                        "mentions",
                        "EXTRACTED",
                        1.0,
                        (item.source_id,),
                    )
                )
        for entity, source_ids in entity_sources.items():
            if len(source_ids) < 2:
                continue
            for left, right in zip(source_ids, source_ids[1:]):
                self.edges.append(
                    GraphEdge(
                        f"source:{left}",
                        f"source:{right}",
                        f"shared_entity:{entity}",
                        "INFERRED",
                        0.7,
                        (left, right),
                    )
                )

    def rank_sources(self, query: str) -> list[tuple[float, ContextItem]]:
        query_terms = terms(query)
        ranked = []
        entity_bonus = defaultdict(float)
        for edge in self.edges:
            if edge.provenance == "INFERRED" and edge.relation.startswith("shared_entity:"):
                entity = edge.relation.split(":", 1)[1].casefold()
                if entity in query.casefold():
                    for source_id in edge.source_ids:
                        entity_bonus[source_id] += edge.confidence
        for item in self.items:
            item_terms = terms(item.text)
            overlap = len(query_terms & item_terms) / max(len(query_terms), 1)
            score = 0.55 * overlap + 0.25 * item.priority + 0.1 * min(item.recency, 3)
            score += 0.1 * entity_bonus[item.source_id]
            ranked.append((score, item))
        return sorted(ranked, key=lambda pair: (-pair[0], pair[1].source_id))

    def subgraph(self, source_ids: set[str]):
        node_ids = {f"source:{source_id}" for source_id in source_ids}
        for edge in self.edges:
            if edge.source in node_ids or edge.target in node_ids:
                node_ids.add(edge.source)
                node_ids.add(edge.target)
        nodes = tuple(node for node in self.nodes if node.node_id in node_ids)
        edges = tuple(
            edge for edge in self.edges
            if edge.source in node_ids and edge.target in node_ids
        )
        return nodes, edges

