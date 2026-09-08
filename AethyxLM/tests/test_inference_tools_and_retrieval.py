import pytest

import chat
from inference.prompt_contract import get_prompt_contract
from inference.retrieval import EvidenceIndex, EvidencePassage, format_evidence
from inference.tools import (
    RestrictedPythonBackend,
    ToolCallError,
    ToolController,
    calculate,
    extract_arithmetic_expression,
    parse_tool_call,
    route_arithmetic_question,
)


def test_calculator_is_correct_and_rejects_non_arithmetic():
    assert calculate("(17 * 23) + 0.5") == "391.5"
    with pytest.raises(ToolCallError):
        calculate("__import__('os').getenv('PATH')")
    with pytest.raises(ToolCallError):
        calculate("2 ** 1000")


def test_tool_schema_is_an_exact_allowlist_and_code_is_default_off():
    with pytest.raises(ToolCallError):
        parse_tool_call({"tool": "shell", "arguments": {"command": "dir"}})
    with pytest.raises(ToolCallError):
        parse_tool_call({"tool": "calculator", "arguments": {"expression": "2+2"}, "extra": 1})
    result = ToolController().execute({
        "tool": "python", "arguments": {"language": "python-restricted", "code": "print(2+2)"}
    })
    assert result["error"] == "disabled"


def test_restricted_python_enforces_timeout_and_output_limit():
    timeout = RestrictedPythonBackend(timeout_seconds=0.1).run("while True:\n    pass")
    assert timeout["error"] == "timeout"
    output = RestrictedPythonBackend(max_output_bytes=8).run("print('123456789')")
    assert output["error"] == "output_limit"
    rejected = ToolController(allow_code=True).execute({
        "tool": "python", "arguments": {"language": "python-restricted", "code": "import os"}
    })
    assert rejected["error"] == "invalid_code"


def test_malformed_calls_have_bounded_retry_signal():
    controller = ToolController(malformed_retries=1)
    assert controller.execute("not json", malformed_attempt=0)["retry_allowed"] is True
    assert controller.execute("not json", malformed_attempt=1)["retry_allowed"] is False
    repaired = controller.execute_with_repair_candidates([
        "not json",
        '{"tool":"calculator","arguments":{"expression":"6*7"}}',
        '{"tool":"calculator","arguments":{"expression":"1+1"}}',
    ])
    assert repaired["result"] == "42"


def test_model_tool_call_protocol_keeps_visible_assistant_prefix():
    executed = ToolController().execute_model_output(
        'I will calculate it. <TOOL_CALL>\n{"tool":"calculator","arguments":{"expression":"7*8"}}'
    )
    assert executed["assistant_text"] == "I will calculate it."
    assert executed["result"]["result"] == "56"


def test_retrieval_preserves_exact_source_text_and_prompt_integration():
    passage = EvidencePassage("invoice-7", "Invoice ID AX-009 totals INR 12,345.67.")
    results = EvidenceIndex([passage, EvidencePassage("other", "Weather is sunny.")]).search("AX-009 total")
    assert results[0].source_id == "invoice-7"
    assert "INR 12,345.67" in format_evidence(results)
    rendered = get_prompt_contract("aethyx-sft-v1").format_tool_result(
        "calculator", {"ok": True, "result": "391"}
    )
    assert "<TOOL_RESULT" in rendered and '"391"' in rendered


@pytest.mark.parametrize(
    ("prompt", "expression", "result"),
    [
        ("What is 17 x 23", "17*23", "391"),
        ("17*23", "17*23", "391"),
        ("calculate (8+2)/5", "(8+2)/5", "2"),
        ("Compute the value of (8.5 + 1.5) / 5?", "(8.5 + 1.5) / 5", "2"),
        ("What is 6 × 7?", "6*7", "42"),
        ("evaluate 2**3 + 4%3", "2**3 + 4%3", "9"),
    ],
)
def test_standalone_arithmetic_is_extracted_and_routed(prompt, expression, result):
    assert extract_arithmetic_expression(prompt) == expression
    routed = route_arithmetic_question(prompt, ToolController())
    assert routed == {"ok": True, "tool": "calculator", "result": result}


@pytest.mark.parametrize(
    "prompt",
    [
        "A room is 17 x 23 cm",
        "What is 17 x 23 cm?",
        "The shipment contains 17 boxes and 23 labels.",
        "Solve x + 2 = 4",
        "2026-09-08",
        "What is 12/09/2026?",
        "123-456-7890",
        "version 1.2.3",
        "https://example.com/17/23",
        "AB-17-23",
    ],
)
def test_automatic_calculator_rejects_ambiguous_non_arithmetic_text(prompt):
    assert extract_arithmetic_expression(prompt) is None
    assert route_arithmetic_question(prompt, ToolController()) is None


@pytest.mark.parametrize("prompt", ["calculate 1/0", "What is 2 +?", "17 x nope"])
def test_invalid_automatic_calculation_falls_through(prompt):
    assert route_arithmetic_question(prompt, ToolController()) is None


class _InteractiveTokenizer:
    eos_id = None

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8")


class _InteractiveModel:
    context_length = 1024


@pytest.mark.parametrize(
    ("mode", "contract_name"),
    [("base", "base-v1"), ("chat", "legacy-chat-v1")],
)
def test_interactive_modes_route_arithmetic_without_model_generation(
    monkeypatch, capsys, mode, contract_name
):
    entries = iter(["What is 17 x 23", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(entries))
    monkeypatch.setattr(
        chat,
        "generate",
        lambda *_args, **_kwargs: pytest.fail("automatic arithmetic reached the model"),
    )

    chat.interactive_session(
        _InteractiveModel(),
        _InteractiveTokenizer(),
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.18,
        no_repeat_ngram_size=4,
        max_new=32,
        stream=False,
        mode=mode,
        prompt_contract=get_prompt_contract(contract_name),
        tool_controller=ToolController(),
    )

    assert "Aethyx: 391" in capsys.readouterr().out


def test_manual_calculator_call_remains_compatible():
    result = ToolController().execute(
        '{"tool":"calculator","arguments":{"expression":"17*23"}}'
    )
    assert result["result"] == "391"
