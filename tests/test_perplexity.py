"""Unit tests for the perplexity input defense (no model downloads)."""

from math import exp
from types import SimpleNamespace

import pytest
import torch

from defenses import get_defense
from defenses.both import BothDefense
from defenses.none import NoneDefense
from defenses.perplexity import (
    HuggingFacePerplexityScorer,
    PerplexityDefense,
    PerplexityResult,
    extract_latest_user_text,
)
from defenses.self_reminder import SelfReminderDefense
from run import evaluate_condition


class FakeTarget:
    def __init__(self):
        self.calls = []

    def chat_with_stats(self, messages):
        self.calls.append(messages)
        return "target response", {
            "latency_s": 0.25,
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
            "tokens_per_second": 8,
        }


class FakeScorer:
    model_name = "fake-gpt2"
    requested_device = "cpu"
    stride = 3

    def __init__(self, result=None, error=None):
        self.result = result or PerplexityResult(10.0, 4, 3, "cpu")
        self.error = error
        self.texts = []

    def score(self, text):
        self.texts.append(text)
        if self.error:
            raise self.error
        return self.result


def make_defense(score=10.0, threshold=10.0, **kwargs):
    target = FakeTarget()
    scorer = kwargs.pop(
        "scorer", FakeScorer(PerplexityResult(score, 5, 4, "cpu"))
    )
    defense = PerplexityDefense(target, scorer=scorer, threshold=threshold, **kwargs)
    return defense, target, scorer


@pytest.mark.parametrize("score", [9.9, 10.0])
def test_score_at_or_below_threshold_allows_and_calls_target_once(score):
    defense, target, _ = make_defense(score=score)

    response, metadata = defense.query([{"role": "user", "content": "hello"}])

    assert response == "target response"
    assert len(target.calls) == 1
    assert metadata["defense_blocked"] is False
    assert metadata["target_invoked"] is True
    assert metadata["target_stats"]["total_tokens"] == 6


def test_score_above_threshold_blocks_without_target_call():
    defense, target, _ = make_defense(score=10.01)

    response, metadata = defense.query([{"role": "user", "content": "hello"}])

    assert response == defense.blocked_response
    assert target.calls == []
    assert metadata["defense_blocked"] is True
    assert metadata["target_invoked"] is False
    assert metadata["target_stats"]["latency_s"] == 0
    assert metadata["target_stats"]["total_tokens"] == 0


def test_latest_user_message_only_is_scored():
    messages = [
        {"role": "system", "content": "system secret"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "assistant secret"},
        {"role": "user", "content": "latest user"},
    ]
    defense, _, scorer = make_defense()

    defense.query(messages)

    assert extract_latest_user_text(messages) == "latest user"
    assert scorer.texts == ["latest user"]


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "x"}]])
def test_missing_user_input_is_neutral(messages):
    scorer = FakeScorer(PerplexityResult(None, 0, 0, "cpu"))
    defense, target, _ = make_defense(scorer=scorer)

    _, metadata = defense.query(messages)

    assert scorer.texts == [""]
    assert metadata["perplexity_score"] is None
    assert metadata["perplexity_measured"] is False
    assert metadata["perplexity_status"] == "no_user_message"
    assert metadata["perplexity_predicted_token_count"] == 0
    assert len(target.calls) == 1


def test_empty_user_input_is_neutral():
    scorer = FakeScorer(PerplexityResult(None, 0, 0, "cpu"))
    defense, target, _ = make_defense(scorer=scorer)

    _, metadata = defense.query([{"role": "user", "content": ""}])

    assert metadata["perplexity_score"] is None
    assert metadata["perplexity_measured"] is False
    assert metadata["perplexity_status"] == "empty_user_message"
    assert metadata["perplexity_token_count"] == 0
    assert len(target.calls) == 1


@pytest.mark.parametrize(
    "message,error_type",
    [({"role": "user"}, ValueError), ({"role": "user", "content": 12}, TypeError)],
)
def test_malformed_user_content_raises_from_extractor(message, error_type):
    with pytest.raises(error_type):
        extract_latest_user_text([message])


@pytest.mark.parametrize("message", ["bad message", {"content": "missing role"}])
def test_malformed_message_uses_failure_policy(message):
    scorer = FakeScorer()
    defense, target, _ = make_defense(scorer=scorer, failure_policy="block")

    _, metadata = defense.query([message])

    assert metadata["perplexity_status"] == "error"
    assert metadata["perplexity_error"]
    assert metadata["defense_blocked"] is True
    assert target.calls == []


def test_unscorable_input_is_not_a_scoring_failure_under_block_policy():
    scorer = FakeScorer(PerplexityResult(None, 1, 0, "cpu"))
    defense, target, _ = make_defense(scorer=scorer, failure_policy="block")

    _, metadata = defense.query([{"role": "user", "content": "x"}])

    assert metadata["perplexity_status"] == "insufficient_tokens"
    assert metadata["perplexity_error"] == ""
    assert metadata["defense_blocked"] is False
    assert len(target.calls) == 1


def test_metadata_preserves_score_configuration_counts_device_and_latency():
    defense, _, _ = make_defense(score=7.5, threshold=11.0)

    _, metadata = defense.query([{"role": "user", "content": "hello"}])

    assert metadata["defense_name"] == "perplexity"
    assert metadata["defense_stage"] == "input"
    assert metadata["perplexity_score"] == 7.5
    assert metadata["perplexity_threshold"] == 11.0
    assert metadata["perplexity_token_count"] == 5
    assert metadata["perplexity_predicted_token_count"] == 4
    assert metadata["perplexity_model"] == "fake-gpt2"
    assert metadata["perplexity_device"] == "cpu"
    assert metadata["perplexity_stride"] == 3
    assert metadata["perplexity_latency_ms"] >= 0
    assert metadata["defense_latency_s"] >= 0
    assert metadata["perplexity_error"] == ""


@pytest.mark.parametrize("policy,blocked,calls", [("allow", False, 1), ("block", True, 0)])
def test_non_raising_failure_policies(policy, blocked, calls):
    scorer = FakeScorer(error=RuntimeError("scorer broke"))
    defense, target, _ = make_defense(scorer=scorer, failure_policy=policy)

    _, metadata = defense.query([{"role": "user", "content": "hello"}])

    assert metadata["defense_blocked"] is blocked
    assert len(target.calls) == calls
    assert metadata["perplexity_score"] is None
    assert metadata["perplexity_error"] == "RuntimeError: scorer broke"
    assert metadata["perplexity_failure_policy"] == policy


def test_raise_failure_policy_wraps_clear_error():
    scorer = FakeScorer(error=ValueError("bad tokens"))
    defense, target, _ = make_defense(scorer=scorer, failure_policy="raise")

    with pytest.raises(RuntimeError, match="Perplexity input defense failed: bad tokens"):
        defense.query([{"role": "user", "content": "hello"}])
    assert target.calls == []


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[int(part) for part in text.split()]])}


class RecordingLossModel:
    config = SimpleNamespace(max_position_embeddings=4, n_positions=4)

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, labels):
        evaluated = labels[:, 1:][labels[:, 1:] != -100].float()
        self.calls.append((input_ids.clone(), labels.clone(), evaluated.clone()))
        return SimpleNamespace(loss=evaluated.mean())


def loaded_scorer(stride=2):
    scorer = HuggingFacePerplexityScorer("fake", "cpu", stride)
    scorer._torch = torch
    scorer._tokenizer = FakeTokenizer()
    scorer._model = RecordingLossModel()
    scorer._resolved_device = "cpu"
    return scorer


def test_one_token_input_has_documented_neutral_score():
    result = loaded_scorer().score("7")

    assert result.score is None
    assert result.token_count == 1
    assert result.predicted_token_count == 0


def test_long_input_aggregates_nll_and_counts_each_prediction_once():
    scorer = loaded_scorer(stride=2)

    result = scorer.score("0 1 2 3 4 5 6")

    evaluated = torch.cat([call[2] for call in scorer._model.calls]).tolist()
    assert evaluated == [1, 2, 3, 4, 5, 6]
    assert result.token_count == 7
    assert result.predicted_token_count == 6
    assert result.score == pytest.approx(exp(sum(range(1, 7)) / 6))


def test_long_input_includes_final_suffix_token():
    scorer = loaded_scorer(stride=3)

    scorer.score("0 1 2 3 4 5 99")

    all_window_ids = torch.cat([call[0].flatten() for call in scorer._model.calls])
    evaluated = torch.cat([call[2] for call in scorer._model.calls])
    assert 99 in all_window_ids.tolist()
    assert evaluated[-1].item() == 99


@pytest.mark.parametrize(
    "tokens,expected",
    [
        ("0 1 2 3", [1, 2, 3]),
        ("0 1 2 3 4", [1, 2, 3, 4]),
    ],
)
def test_context_length_boundaries_score_every_predictable_token(tokens, expected):
    scorer = loaded_scorer(stride=2)

    result = scorer.score(tokens)

    evaluated = torch.cat([call[2] for call in scorer._model.calls]).tolist()
    assert evaluated == expected
    assert result.predicted_token_count == len(expected)


def test_stride_larger_than_usable_context_is_rejected():
    scorer = loaded_scorer(stride=4)

    with pytest.raises(ValueError, match="stride 4 exceeds the maximum 3"):
        scorer.score("0 1")


def test_tokenizer_sentinel_does_not_override_model_context():
    scorer = loaded_scorer()
    scorer._tokenizer.model_max_length = 10**30

    assert scorer._context_length() == 4


class Availability:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=Availability(cuda),
        backends=SimpleNamespace(mps=Availability(mps)),
    )


@pytest.mark.parametrize(
    "torch_module,expected",
    [
        (fake_torch(cuda=True, mps=True), "cuda"),
        (fake_torch(cuda=False, mps=True), "mps"),
        (fake_torch(cuda=False, mps=False), "cpu"),
    ],
)
def test_auto_device_priority(torch_module, expected):
    assert HuggingFacePerplexityScorer._resolve_device(torch_module, "auto") == expected


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_has_clear_error(device):
    with pytest.raises(RuntimeError, match=f"'{device}'.*unavailable"):
        HuggingFacePerplexityScorer._resolve_device(fake_torch(), device)


def test_scorer_is_lazy_and_reuses_loaded_objects():
    scorer = HuggingFacePerplexityScorer("fake", "cpu", 2)
    assert scorer._model is None
    model = RecordingLossModel()
    scorer._torch = torch
    scorer._tokenizer = FakeTokenizer()
    scorer._model = model
    scorer._resolved_device = "cpu"

    scorer.score("0 1 2")
    scorer.score("0 1 2")

    assert scorer._model is model
    assert len(model.calls) == 2


def test_target_exception_is_not_converted_to_a_perplexity_failure():
    class FailingTarget(FakeTarget):
        def chat_with_stats(self, messages):
            raise LookupError("target failed")

    defense = PerplexityDefense(
        FailingTarget(),
        scorer=FakeScorer(PerplexityResult(2.0, 3, 2, "cpu")),
        threshold=10,
        failure_policy="allow",
    )

    with pytest.raises(LookupError, match="target failed"):
        defense.query([{"role": "user", "content": "hello"}])


def test_registration_and_existing_defenses_still_resolve():
    target = FakeTarget()

    perplexity = get_defense("perplexity", target)
    assert isinstance(perplexity, PerplexityDefense)
    assert perplexity.scorer._model is None
    assert isinstance(get_defense("none", target), NoneDefense)
    assert isinstance(get_defense("self_reminder", target), SelfReminderDefense)
    assert isinstance(get_defense("both", target, FakeTarget()), BothDefense)


def test_runner_preserves_block_metadata_and_accepts_zero_target_stats():
    defense, _, _ = make_defense(score=10.01)

    result = evaluate_condition(
        [{"role": "user", "content": "blocked"}], defense
    )

    assert result["defense_name"] == "perplexity"
    assert result["defense_blocked"] is True
    assert result["target_invoked"] is False
    assert result["target_latency_s"] == 0
    assert result["target_total_tokens"] == 0
    assert result["perplexity_score"] == 10.01
    assert result["total_latency_s"] >= 0
