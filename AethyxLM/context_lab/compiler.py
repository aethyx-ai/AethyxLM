"""Local hybrid compiler with graph retrieval and conservative visual gating."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional

from context_lab.graph import DeterministicContextGraph
from context_lab.risk import assess_verbatim_risk
from context_lab.schema import CompiledContext, ContextRequest
from context_lab.visual import plan_visual_pages


@dataclass(frozen=True)
class CompilerPolicy:
    mode: str = "graph"
    max_selected_items: int = 12
    recent_turns_to_keep: int = 2
    visual_min_chars: int = 6000
    visual_chars_per_page: int = 24000
    target_model_supports_vision: bool = False
    minimum_estimated_reduction: float = 0.05

    def __post_init__(self):
        if self.mode not in {"graph", "visual", "hybrid"}:
            raise ValueError("mode must be graph, visual, or hybrid")
        if self.max_selected_items <= 0:
            raise ValueError("max_selected_items must be positive")
        if not 0 <= self.minimum_estimated_reduction < 1:
            raise ValueError("minimum_estimated_reduction must be within [0, 1)")


class LocalContextCompiler:
    """Compile context locally while preserving exact-risk and provenance data."""

    VERSION = "aethyx-context/0.1"

    def __init__(
        self,
        policy: CompilerPolicy | None = None,
        token_counter: Optional[Callable[[str], list[int]]] = None,
    ):
        self.policy = policy or CompilerPolicy()
        self.token_counter = token_counter

    def compile(self, request: ContextRequest) -> CompiledContext:
        recent_conversations = sorted(
            (item for item in request.items if item.kind == "conversation"),
            key=lambda item: (-item.recency, item.source_id),
        )[: self.policy.recent_turns_to_keep]
        recent_ids = {item.source_id for item in recent_conversations}
        protected = []
        compressible = []
        for item in request.items:
            assessment = assess_verbatim_risk(item)
            if assessment.must_preserve or item.source_id in recent_ids:
                protected.append(item)
            else:
                compressible.append(item)

        graph = DeterministicContextGraph(tuple(compressible))
        ranked = graph.rank_sources(request.query)
        selected = [item for _, item in ranked[: self.policy.max_selected_items]]
        selected_ids = {item.source_id for item in selected}
        nodes, edges = graph.subgraph(selected_ids)

        warnings = []
        pages = ()
        visual_candidates = [
            item for item in selected
            if len(item.text) >= self.policy.visual_min_chars
        ]
        if self.policy.mode in {"visual", "hybrid"} and visual_candidates:
            if self.policy.target_model_supports_vision:
                pages = plan_visual_pages(
                    visual_candidates,
                    chars_per_page=self.policy.visual_chars_per_page,
                )
                selected = [
                    item for item in selected if item not in visual_candidates
                ]
            else:
                warnings.append(
                    "Visual candidates were not rendered because the target model "
                    "does not declare a vision input contract."
                )

        canonical_sources = json.dumps(
            [(item.source_id, item.kind, item.text) for item in request.items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_sources.encode("utf-8")).hexdigest()
        represented = {item.source_id for item in protected + selected}
        represented.update(source_id for page in pages for source_id in page.source_ids)
        compiled = CompiledContext(
            version=self.VERSION,
            mode=self.policy.mode,
            query=request.query,
            protected_items=tuple(protected),
            selected_items=tuple(selected),
            graph_nodes=nodes,
            graph_edges=edges,
            visual_pages=pages,
            omitted_source_ids=tuple(
                item.source_id for item in request.items
                if item.source_id not in represented
            ),
            source_digest=digest,
            warnings=tuple(warnings),
        )
        if self.token_counter is not None and not compiled.visual_pages:
            raw_text = request.query + "\n" + "\n".join(
                item.text for item in request.items
            )
            compact_text = json.dumps(
                compiled.model_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            raw_units = len(self.token_counter(raw_text))
            compact_units = len(self.token_counter(compact_text))
            required = raw_units * (1 - self.policy.minimum_estimated_reduction)
            if compact_units >= required:
                return CompiledContext(
                    version=self.VERSION,
                    mode="raw",
                    query=request.query,
                    protected_items=tuple(request.items),
                    selected_items=(),
                    graph_nodes=(),
                    graph_edges=(),
                    visual_pages=(),
                    omitted_source_ids=(),
                    source_digest=digest,
                    warnings=tuple(warnings) + (
                        "Compression bypassed because the compact payload did not "
                        "meet the configured profitability threshold.",
                    ),
                )
        return compiled
