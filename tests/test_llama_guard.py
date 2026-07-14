"""Unit tests for native-conversation Llama Guard filtering (no Ollama)."""

import pytest

from defenses import defense_requires_guard, get_defense
from defenses.both import BothDefense
from defenses.llama_guard import (
    LlamaGuardDefense,
    build_input_guard_conversation,
    build_output_guard_conversation,
    parse_guard_output,
)
from defenses.none import NoneDefense
from defenses.perplexity import PerplexityDefense
from defenses.self_reminder import SelfReminderDefense
from model_client import ModelClient
from run import evaluate_condition


STATS = {
    "latency_s": 0.2,
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12,
    "tokens_per_second": 10,
}

REAL_UNSAFE_DEFAMATION_OUTPUT = "unsafe\nS5"
REAL_SAFE_OUTPUT = "safe"


class FakeClient:
    def __init__(self, responses=None, model="fake-model", error=None):
        self.responses = list(responses or [])
        self.model = model
        self.error = error
        self.calls = []

    def chat_with_stats(self, messages, options=None):
        self.calls.append({"messages": messages, "options": options})
        if self.error:
            raise self.error
        response = self.responses.pop(0) if self.responses else "target response"
        return response, dict(STATS)


def make_defense(mode, guard_responses, policy="allow", target=None, guard=None):
    target = target or FakeClient(["exact target response"], "target")
    guard = guard or FakeClient(guard_responses, "llama-guard3:1b")
    return LlamaGuardDefense(target, guard, mode=mode, failure_policy=policy), target, guard


@pytest.mark.parametrize(
    "raw,label,categories",
    [
        ("safe", "safe", []),
        ("SAFE", "safe", []),
        (" unsafe ", "unsafe", []),
        ("unsafe\nS1", "unsafe", ["S1"]),
        ("unsafe\nS1,S2", "unsafe", ["S1", "S2"]),
        ("unsafe\nS1\nS2", "unsafe", ["S1", "S2"]),
        ("unsafe\nS1,S1,S13", "unsafe", ["S1", "S13"]),
        ("\n unsafe\n s13 \n", "unsafe", ["S13"]),
    ],
)
def test_parse_valid_outputs(raw, label, categories):
    result = parse_guard_output(raw)

    assert result.label == label
    assert result.categories == categories
    assert result.raw_output == raw
    assert result.parse_status == "parsed"
    assert result.error == ""


def test_real_ollama_output_fixtures():
    unsafe = parse_guard_output(REAL_UNSAFE_DEFAMATION_OUTPUT)
    safe = parse_guard_output(REAL_SAFE_OUTPUT)

    assert unsafe.label == "unsafe"
    assert unsafe.categories == ["S5"]
    assert unsafe.raw_output == "unsafe\nS5"
    assert safe.label == "safe"
    assert safe.categories == []


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "moderation result: safe",
        "S1",
        "this is unsafe",
        "safety",
        "unsafe\nS0",
        "unsafe\nS14",
        "safe\nS1",
        "unsafe\nS1 and S2",
    ],
)
def test_parse_rejects_malformed_outputs(raw):
    result = parse_guard_output(raw)

    assert result.label == "unknown"
    assert result.categories == []
    assert result.parse_status == "malformed"
    assert result.error
    assert result.raw_output == raw


def test_input_conversation_ends_at_latest_user_without_mutation():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "ignored later answer"},
    ]
    original = [dict(message) for message in messages]

    conversation, status = build_input_guard_conversation(messages)

    assert status == "classifiable"
    assert conversation[-1] == {"role": "user", "content": "latest"}
    assert len(conversation) == 4
    assert messages == original
    assert conversation is not messages


def test_output_conversation_appends_one_assistant_without_mutation():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "existing history"},
    ]
    original = [dict(message) for message in messages]

    conversation, status = build_output_guard_conversation(messages, "")

    assert status == "classifiable"
    assert conversation == messages + [{"role": "assistant", "content": ""}]
    assert messages == original


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "x"}]])
def test_no_user_conversation_is_neutral_unclassifiable(messages):
    conversation, status = build_input_guard_conversation(messages)

    assert conversation == []
    assert status == "no_user_message"


@pytest.mark.parametrize(
    "policy,blocked,target_calls",
    [("allow", False, 1), ("block", True, 0)],
)
def test_unclassifiable_input_is_distinct_and_follows_policy(
    policy, blocked, target_calls
):
    defense, target, guard = make_defense("input", [], policy=policy)

    _, meta = defense.query([])

    assert len(target.calls) == target_calls
    assert guard.calls == []
    assert meta["defense_blocked"] is blocked
    assert meta["input_guard"]["parse_status"] == "unclassifiable"
    assert meta["input_guard"]["error"] == ""


@pytest.mark.parametrize(
    "messages",
    [["bad"], [{"content": "missing role"}], [{"role": "user", "content": 3}]],
)
def test_malformed_conversation_is_rejected(messages):
    with pytest.raises((TypeError, ValueError)):
        build_input_guard_conversation(messages)


def test_safe_input_calls_guard_then_target_once_with_native_messages():
    messages = [{"role": "user", "content": "hello"}]
    defense, target, guard = make_defense("input", ["safe"])

    response, meta = defense.query(messages)

    assert response == "exact target response"
    assert len(target.calls) == 1
    assert guard.calls == [{"messages": messages, "options": {"temperature": 0}}]
    assert meta["input_guard"]["label"] == "safe"
    assert meta["input_guard"]["latency_s"] == 0.2
    assert meta["target_invoked"] is True
    assert meta["defense_latency_s"] == pytest.approx(0.2)
    assert meta["guard_stats"]["tokens_per_second"] == 10


def test_unsafe_input_blocks_without_target_and_preserves_metadata():
    defense, target, guard = make_defense("input", [REAL_UNSAFE_DEFAMATION_OUTPUT])

    response, meta = defense.query([{"role": "user", "content": "bad"}])

    assert response == defense.blocked_response
    assert target.calls == []
    assert len(guard.calls) == 1
    assert meta["defense_blocked"] is True
    assert meta["target_invoked"] is False
    assert meta["response_replaced"] is False
    assert meta["input_guard"]["categories"] == ["S5"]
    assert meta["target_stats"]["total_tokens"] == 0
    assert meta["defense_latency_s"] == pytest.approx(0.2)
    assert meta["guard_stats"]["tokens_per_second"] == 10


def test_safe_output_returns_exact_target_and_appends_it_once():
    messages = [{"role": "user", "content": "hello"}]
    defense, target, guard = make_defense("output", ["safe"])

    response, meta = defense.query(messages)

    assert response == "exact target response"
    assert len(target.calls) == 1
    assert guard.calls[0]["messages"] == messages + [
        {"role": "assistant", "content": "exact target response"}
    ]
    assert meta["target_invoked"] is True
    assert meta["response_replaced"] is False
    assert meta["output_guard"]["label"] == "safe"
    assert meta["defense_latency_s"] == pytest.approx(0.2)
    assert meta["guard_stats"]["tokens_per_second"] == 10


def test_empty_target_response_is_classified_and_preserved_when_safe():
    target = FakeClient([""], "target")
    defense, _, guard = make_defense("output", ["safe"], target=target)

    response, meta = defense.query([{"role": "user", "content": "hello"}])

    assert response == ""
    assert guard.calls[0]["messages"][-1] == {"role": "assistant", "content": ""}
    assert meta["response_replaced"] is False


def test_unsafe_output_replaces_response_after_one_target_call():
    defense, target, _ = make_defense("output", ["unsafe\nS2"])

    response, meta = defense.query([{"role": "user", "content": "hello"}])

    assert response == defense.blocked_response
    assert len(target.calls) == 1
    assert meta["target_invoked"] is True
    assert meta["defense_blocked"] is True
    assert meta["response_replaced"] is True
    assert meta["original_response"] == "exact target response"


def test_both_unsafe_input_short_circuits():
    defense, target, guard = make_defense("both", ["unsafe\nS2", "safe"])

    _, meta = defense.query([{"role": "user", "content": "hello"}])

    assert target.calls == []
    assert len(guard.calls) == 1
    assert meta["input_guard"]["label"] == "unsafe"
    assert meta["output_guard"] is None
    assert meta["defense_latency_s"] == pytest.approx(0.2)
    assert meta["guard_stats"]["tokens_per_second"] == 10


def test_both_safe_input_checks_output_and_aggregates_guard_stats():
    defense, target, guard = make_defense("both", ["safe", "unsafe\nS3"])

    response, meta = defense.query([{"role": "user", "content": "hello"}])

    assert response == defense.blocked_response
    assert len(target.calls) == 1
    assert len(guard.calls) == 2
    assert meta["input_guard"]["label"] == "safe"
    assert meta["output_guard"]["label"] == "unsafe"
    assert meta["guard_total_latency_s"] == pytest.approx(0.4)
    assert meta["guard_stats"]["prompt_tokens"] == 20
    assert meta["guard_stats"]["completion_tokens"] == 4
    assert meta["defense_latency_s"] == pytest.approx(0.4)
    assert meta["guard_stats"]["tokens_per_second"] == pytest.approx(10)


@pytest.mark.parametrize(
    "stage,policy,expected_blocked,target_calls",
    [
        ("input", "allow", False, 1),
        ("input", "block", True, 0),
        ("output", "allow", False, 1),
        ("output", "block", True, 1),
    ],
)
def test_parse_failure_policies(stage, policy, expected_blocked, target_calls):
    defense, target, _ = make_defense(stage, ["not a label"], policy=policy)

    response, meta = defense.query([{"role": "user", "content": "hello"}])

    assert len(target.calls) == target_calls
    assert meta["defense_blocked"] is expected_blocked
    if expected_blocked:
        assert response == defense.blocked_response
    else:
        assert response == "exact target response"
    assert meta["defense_latency_s"] == pytest.approx(0.2)
    assert meta["guard_stats"]["tokens_per_second"] == 10


@pytest.mark.parametrize("stage", ["input", "output"])
def test_raise_policy_for_parse_failure(stage):
    defense, target, _ = make_defense(stage, ["bad"], policy="raise")

    with pytest.raises(RuntimeError, match=f"Llama Guard {stage} classification failed"):
        defense.query([{"role": "user", "content": "hello"}])
    assert len(target.calls) == (1 if stage == "output" else 0)


@pytest.mark.parametrize(
    "stage,policy,target_calls,blocked",
    [
        ("input", "allow", 1, False),
        ("input", "block", 0, True),
        ("output", "allow", 1, False),
        ("output", "block", 1, True),
    ],
)
def test_provider_failure_policies(stage, policy, target_calls, blocked):
    guard = FakeClient(model="guard", error=ConnectionError("offline"))
    defense, target, _ = make_defense(stage, [], policy=policy, guard=guard)

    _, meta = defense.query([{"role": "user", "content": "hello"}])

    assert len(target.calls) == target_calls
    assert meta["defense_blocked"] is blocked
    decision = meta["input_guard"] if stage == "input" else meta["output_guard"]
    assert decision["parse_status"] == "provider_error"
    assert "ConnectionError" in decision["error"]
    assert meta["defense_latency_s"] >= 0
    assert meta["defense_latency_s"] == meta["guard_total_latency_s"]
    assert meta["guard_stats"]["tokens_per_second"] == 0


@pytest.mark.parametrize("stage", ["input", "output"])
def test_provider_failure_raise_policy(stage):
    guard = FakeClient(model="guard", error=ConnectionError("offline"))
    defense, target, _ = make_defense(
        stage, [], policy="raise", guard=guard
    )

    with pytest.raises(RuntimeError, match=f"Llama Guard {stage} classification failed"):
        defense.query([{"role": "user", "content": "hello"}])
    assert len(target.calls) == (1 if stage == "output" else 0)


def test_target_error_is_not_guard_error():
    target = FakeClient(error=LookupError("target failed"))
    defense, _, guard = make_defense("output", ["safe"], target=target)

    with pytest.raises(LookupError, match="target failed"):
        defense.query([{"role": "user", "content": "hello"}])
    assert guard.calls == []


def test_model_client_forwards_deterministic_options():
    class RawResponse:
        prompt_eval_count = 1
        eval_count = 1
        eval_duration = 1_000_000_000
        message = type("Message", (), {"content": "safe"})()

    class RawClient:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return RawResponse()

    client = ModelClient.__new__(ModelClient)
    client.model = "guard"
    client._client = RawClient()

    client.chat_with_stats(
        [{"role": "user", "content": "hello"}], options={"temperature": 0}
    )

    assert client._client.kwargs["options"] == {"temperature": 0}


def test_registration_modes_and_existing_registrations():
    target = FakeClient()
    guard = FakeClient()

    assert get_defense("llama_guard_input", target, guard).mode == "input"
    assert get_defense("llama_guard", target, guard).mode == "output"
    assert get_defense("llama_guard_both", target, guard).mode == "both"
    assert defense_requires_guard("llama_guard_input") is True
    assert isinstance(get_defense("none", target), NoneDefense)
    assert isinstance(get_defense("self_reminder", target), SelfReminderDefense)
    assert isinstance(get_defense("both", target, guard), BothDefense)
    assert isinstance(get_defense("perplexity", target), PerplexityDefense)


@pytest.mark.parametrize(
    "mode,guard_output,expected_invoked,expected_replaced",
    [("input", "unsafe\nS1", False, False), ("output", "unsafe\nS1", True, True)],
)
def test_runner_accepts_guard_block_metadata(
    mode, guard_output, expected_invoked, expected_replaced
):
    defense, _, _ = make_defense(mode, [guard_output])

    result = evaluate_condition([{"role": "user", "content": "hello"}], defense)

    assert result["target_invoked"] is expected_invoked
    assert result["response_replaced"] is expected_replaced
    assert result["llama_guard_mode"] == mode
    assert result["guard_total_latency_s"] == pytest.approx(0.2)
    assert result["defense_latency_s"] == pytest.approx(0.2)
    assert result["guard_tokens_per_second"] == 10
    expected_target_latency = 0 if mode == "input" else 0.2
    assert result["target_latency_s"] == pytest.approx(expected_target_latency)
    assert result["total_latency_s"] == pytest.approx(
        expected_target_latency + result["defense_latency_s"]
    )
