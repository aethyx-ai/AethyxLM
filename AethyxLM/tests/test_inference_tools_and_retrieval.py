from types import SimpleNamespace

import pytest

import chat
from inference.prompt_contract import get_prompt_contract
from inference.retrieval import EvidenceIndex, EvidencePassage, format_evidence
from inference.tools import (
    RestrictedPythonBackend,
    ToolCallError,
    ToolController,
    parse_tool_call,
)


def test_tool_schema_is_an_exact_allowlist_and_code_is_default_off():
    with pytest.raises(ToolCallError):
        parse_tool_call({"tool": "shell", "arguments": {"command": "dir"}})
    with pytest.raises(ToolCallError):
        parse_tool_call(
            {
                "tool": "python",
                "arguments": {
                    "language": "python-restricted",
                    "code": "print(4)",
                    "extra": True,
                },
            }
        )
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
    controller = ToolController(allow_code=True, malformed_retries=1)
    assert controller.execute("not json", malformed_attempt=0)["retry_allowed"] is True
    assert controller.execute("not json", malformed_attempt=1)["retry_allowed"] is False
    repaired = controller.execute_with_repair_candidates([
        "not json",
        '{"tool":"python","arguments":{"language":"python-restricted","code":"print(6*7)"}}',
        '{"tool":"python","arguments":{"language":"python-restricted","code":"print(1+1)"}}',
    ])
    assert repaired["stdout"] == "42\n"


def test_model_tool_call_protocol_keeps_visible_assistant_prefix():
    executed = ToolController(allow_code=True).execute_model_output(
        'I will run it. <TOOL_CALL>\n'
        '{"tool":"python","arguments":{"language":"python-restricted","code":"print(7*8)"}}'
    )
    assert executed["assistant_text"] == "I will run it."
    assert executed["result"]["stdout"] == "56\n"


def test_retrieval_preserves_exact_source_text_and_prompt_integration():
    passage = EvidencePassage("invoice-7", "Invoice ID AX-009 totals INR 12,345.67.")
    results = EvidenceIndex([passage, EvidencePassage("other", "Weather is sunny.")]).search("AX-009 total")
    assert results[0].source_id == "invoice-7"
    assert "INR 12,345.67" in format_evidence(results)
    rendered = get_prompt_contract("aethyx-sft-v1").format_tool_result(
        "python", {"ok": True, "stdout": "55\n"}
    )
    assert "<TOOL_RESULT" in rendered and '"55\\n"' in rendered


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
def test_interactive_modes_send_arithmetic_to_model_generation(
    monkeypatch, capsys, mode, contract_name
):
    entries = iter(["What is 17 x 23", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(entries))
    prompts = []

    def fake_generate(_model, _tokenizer, prompt, **_kwargs):
        prompts.append(prompt)
        return SimpleNamespace(text="model answer", dropped_prompt_tokens=0)

    monkeypatch.setattr(chat, "generate", fake_generate)

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
        tool_controller=ToolController(allow_code=True),
    )

    assert prompts and "What is 17 x 23" in prompts[0]
    assert "Aethyx: model answer" in capsys.readouterr().out


def test_manual_restricted_python_call_remains_compatible():
    result = ToolController(allow_code=True).execute(
        '{"tool":"python","arguments":{"language":"python-restricted",'
        '"code":"print(sum(range(1, 11)))"}}'
    )
    assert result["stdout"] == "55\n"
