from context_lab.benchmark import BenchmarkCase, benchmark_compiler
from context_lab.compiler import CompilerPolicy, LocalContextCompiler
from context_lab.schema import ContextItem, ContextRequest
from tokenizer.tokenizer import AethyxTokenizer


def sample_request():
    return ContextRequest(
        request_id="case-1",
        query="Which city is the deployment located in?",
        items=(
            ContextItem(
                "system",
                "system",
                "Follow the user instructions exactly.",
                protected=True,
            ),
            ContextItem(
                "deploy",
                "document",
                "The Aethyx deployment is located in Bengaluru, India.",
                priority=0.9,
            ),
            ContextItem(
                "noise",
                "document",
                "A long unrelated note about gardening and rainfall.",
            ),
            ContextItem(
                "state",
                "agent_state",
                "request_id=123456789 and phase=research",
            ),
        ),
    )


def test_context_compiler_preserves_risky_text_and_provenance():
    compiler = LocalContextCompiler(CompilerPolicy(max_selected_items=1))
    result = compiler.compile(sample_request())

    assert {item.source_id for item in result.protected_items} == {"system", "state"}
    assert result.selected_items[0].source_id == "deploy"
    assert all(edge.provenance in {"EXTRACTED", "INFERRED"} for edge in result.graph_edges)
    assert "noise" in result.omitted_source_ids


def test_context_benchmark_reports_recall_and_no_unverified_accuracy_claim():
    compiler = LocalContextCompiler(CompilerPolicy(max_selected_items=1))
    result = benchmark_compiler(
        compiler,
        AethyxTokenizer(),
        [BenchmarkCase(sample_request(), ("deploy",), ("123456789",))],
    )

    assert result["relevant_source_recall"] == 1.0
    assert result["exact_value_retention"] == 1.0
    assert result["model_task_accuracy"] is None


def test_visual_mode_refuses_to_assume_unsupported_model_capability():
    long_item = ContextItem("bulk", "document", "context " * 2000)
    request = ContextRequest("context", (long_item,))
    result = LocalContextCompiler(
        CompilerPolicy(mode="visual", visual_min_chars=100, target_model_supports_vision=False)
    ).compile(request)

    assert not result.visual_pages
    assert result.warnings


def test_profitability_gate_compresses_only_when_payload_is_smaller():
    tokenizer = AethyxTokenizer()
    request = ContextRequest(
        "Where is the deployment?",
        (
            ContextItem("answer", "document", "The deployment is in Bengaluru.", priority=1.0),
            ContextItem("bulk", "document", "Archived unrelated telemetry. " * 2000),
        ),
    )
    compiler = LocalContextCompiler(
        CompilerPolicy(max_selected_items=1), token_counter=tokenizer.encode
    )

    result = compiler.compile(request)

    assert result.mode == "graph"
    assert result.selected_items[0].source_id == "answer"
    assert "bulk" in result.omitted_source_ids
