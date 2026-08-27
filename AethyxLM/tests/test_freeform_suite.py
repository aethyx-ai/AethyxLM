from evaluation.freeform_suite import (
    FreeFormCase,
    format_case_prompt,
    normalize_answer,
    score_generated_answer,
)
from inference.generation import CHAT_STOP_STRINGS, stop_strings_for_mode


def test_normalize_answer_is_unicode_aware_and_ignores_terminal_punctuation():
    assert normalize_answer("  NEW Delhi. ") == "new delhi"
    assert normalize_answer("１２") == "12"


def test_freeform_scoring_keeps_strict_prefix_and_containment_separate():
    prefix = score_generated_answer("New Delhi is India's capital.", ("New Delhi",))
    assert not prefix["exact_match"]
    assert prefix["answer_prefix_match"]
    assert prefix["answer_contained"]

    buried = score_generated_answer("The answer may be New Delhi", ("New Delhi",))
    assert not buried["exact_match"]
    assert not buried["answer_prefix_match"]
    assert buried["answer_contained"]


def test_case_prompt_does_not_leak_accepted_answers():
    case = FreeFormCase("knowledge", "The capital is", "What is the capital?", ("Answer",))
    assert format_case_prompt(case, "base") == "The capital is"
    assert format_case_prompt(case, "chat") == "User: What is the capital?\nAethyx:"


def test_chat_stop_contract_is_complete_and_centralized():
    assert stop_strings_for_mode("chat") == CHAT_STOP_STRINGS
    assert "User:" in CHAT_STOP_STRINGS
    assert "Aethyx:" in CHAT_STOP_STRINGS
    assert stop_strings_for_mode("base") == ()
