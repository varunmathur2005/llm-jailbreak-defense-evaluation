"""Tests for calibration and local comparison utilities without real models."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.calibrate_perplexity import (
    build_threshold_sweep,
    load_labelled_prompts,
    score_prompts,
    select_thresholds,
    threshold_metrics,
)
from scripts.compare_varun_defenses import (
    ComparisonModelClient,
    aggregate_results,
    evaluate_prompt,
    latency_summary,
    load_prompt_jsonl,
    load_raw_results,
    percentile,
    save_outputs,
)


class FakeScorer:
    model_name = "fake-gpt2"
    requested_device = "cpu"
    stride = 4

    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    def score(self, _prompt):
        self.calls += 1
        return SimpleNamespace(
            score=next(self.scores),
            token_count=5,
            predicted_token_count=4,
            device="cpu",
        )


class FakeDefense:
    def __init__(self, response, metadata):
        self.response = response
        self.metadata = metadata
        self.calls = 0

    def query(self, _messages):
        self.calls += 1
        return self.response, self.metadata


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repository_calibration_and_test_fixtures_are_separate():
    calibration = load_labelled_prompts(Path("data/perplexity_calibration.jsonl"))
    benign_test = load_prompt_jsonl(Path("data/benign_test.jsonl"), "test")
    normalize = lambda text: re.sub(r"\W+", " ", text.casefold()).strip()

    assert len([row for row in calibration if row["label"] == "benign"]) == 20
    assert len([row for row in calibration if row["label"] == "attack"]) == 12
    assert len(benign_test) == 20
    assert {row["id"] for row in calibration}.isdisjoint(
        {row["id"] for row in benign_test}
    )
    assert {normalize(row["prompt"]) for row in calibration}.isdisjoint(
        {normalize(row["prompt"]) for row in benign_test}
    )
    assert all(
        SequenceMatcher(None, normalize(left["prompt"]), normalize(right["prompt"])).ratio()
        < 0.65
        for left in calibration
        for right in benign_test
    )


def test_comparison_target_client_forwards_identical_generation_settings():
    class RawResponse:
        prompt_eval_count = 1
        eval_count = 1
        eval_duration = 1_000_000_000
        message = type("Message", (), {"content": "ok"})()

    class RawClient:
        def chat(self, **kwargs):
            self.kwargs = kwargs
            return RawResponse()

    client = ComparisonModelClient.__new__(ComparisonModelClient)
    client.model = "target"
    client._client = RawClient()
    client._comparison_options = {"temperature": 0.0, "seed": 453}

    client.chat_with_stats([{"role": "user", "content": "hello"}])

    assert client._client.kwargs["options"] == {"temperature": 0.0, "seed": 453}


def test_calibration_rejects_test_split(tmp_path):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [{"id": "x", "split": "test", "label": "benign", "prompt": "hi"}])

    with pytest.raises(ValueError, match="final-test data cannot be used"):
        load_labelled_prompts(path)


def test_benign_test_loader_requires_test_split(tmp_path):
    path = tmp_path / "benign.jsonl"
    write_jsonl(path, [{"id": "x", "split": "calibration", "label": "benign", "prompt": "hi"}])

    with pytest.raises(ValueError, match="split='test'"):
        load_prompt_jsonl(path, "test")


def test_threshold_metrics_and_selection():
    scores = [
        {"label": "benign", "score": 10.0},
        {"label": "benign", "score": 20.0},
        {"label": "attack", "score": 30.0},
        {"label": "attack", "score": 40.0},
    ]

    metrics = threshold_metrics(scores, 20.0)
    sweep = build_threshold_sweep(scores)
    selected = select_thresholds(sweep)

    assert metrics["threshold"] == 20.0
    assert metrics["benign_false_positive_rate"] == 0.0
    assert metrics["attack_detection_rate"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["benign_measured_count"] == 2
    assert metrics["attack_measured_count"] == 2
    assert selected["recommended"]["requested_false_positive_rate"] == 0.05
    assert selected["recommended"]["achieved_false_positive_rate"] <= 0.05
    assert selected["recommended"]["absolute_rate_difference"] >= 0
    assert selected["recommended"]["attack_detection_rate"] == 1.0


def test_twenty_benign_fpr_granularity_and_budgeted_selection():
    scores = [
        *({"label": "benign", "score": float(value)} for value in range(1, 21)),
        {"label": "attack", "score": 100.0},
    ]
    selected = select_thresholds(build_threshold_sweep(scores))

    assert selected["1pct"]["achieved_false_positive_rate"] == 0.0
    assert selected["1pct"]["absolute_rate_difference"] == 0.01
    assert selected["5pct"]["achieved_false_positive_rate"] == 0.05
    assert selected["5pct"]["false_positives"] == 1
    assert selected["5pct"]["benign_measured_count"] == 20
    assert selected["10pct"]["achieved_false_positive_rate"] == 0.10
    assert selected["10pct"]["false_positives"] == 2


def test_strict_threshold_boundaries_include_block_all_and_none():
    scores = [
        {"label": "benign", "score": 10.0},
        {"label": "attack", "score": 20.0},
    ]
    sweep = build_threshold_sweep(scores)

    assert sweep[0]["false_positives"] == 1
    assert sweep[0]["true_positives"] == 1
    at_ten = next(row for row in sweep if row["threshold"] == 10.0)
    assert at_ten["false_positives"] == 0  # Equality is allowed.
    assert sweep[-1]["false_positives"] == 0
    assert sweep[-1]["true_positives"] == 0


def test_score_cache_is_reused(tmp_path):
    records = [
        {"id": "b", "label": "benign", "split": "calibration", "prompt": "hello"},
        {"id": "a", "label": "attack", "split": "calibration", "prompt": "noise"},
    ]
    cache = tmp_path / "cache.json"
    first = FakeScorer([10.0, 100.0])
    scores, computed = score_prompts(records, first, cache)
    second = FakeScorer([])
    cached_scores, cached_computed = score_prompts(records, second, cache)

    assert computed == 2
    assert first.calls == 2
    assert cached_computed == 0
    assert second.calls == 0
    assert cached_scores == scores


@pytest.mark.parametrize("change", ["prompt", "model", "stride", "device"])
def test_cache_invalidates_relevant_configuration_changes(tmp_path, change):
    record = {"id": "x", "label": "benign", "split": "calibration", "prompt": "hello"}
    cache = tmp_path / "cache.json"
    first = FakeScorer([10.0])
    score_prompts([record], first, cache)
    changed_record = dict(record)
    second = FakeScorer([20.0])
    if change == "prompt":
        changed_record["prompt"] = "changed"
    elif change == "model":
        second.model_name = "different-model"
    elif change == "stride":
        second.stride = 8
    else:
        second.requested_device = "mps"

    scores, computed = score_prompts([changed_record], second, cache)

    assert computed == 1
    assert second.calls == 1
    assert scores[0]["score"] == 20.0


def test_duplicate_ids_and_malformed_jsonl_fail_clearly(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    row = {"id": "x", "split": "calibration", "label": "benign", "prompt": "one"}
    write_jsonl(duplicate, [row, {**row, "label": "attack", "prompt": "two"}])
    with pytest.raises(ValueError, match="duplicate id"):
        load_labelled_prompts(duplicate)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"id":', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_labelled_prompts(malformed)


def test_corrupt_cache_fails_safely(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text('{"version": 1, "scores": []}', encoding="utf-8")
    scorer = FakeScorer([10.0])
    record = {"id": "x", "label": "benign", "split": "calibration", "prompt": "hello"}

    with pytest.raises(ValueError, match="scores must be an object"):
        score_prompts([record], scorer, cache)


def target_stats(latency=2.0):
    return {"latency_s": latency, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.mark.parametrize(
    "metadata,response,expected_success",
    [
        ({"defense_blocked": True, "target_invoked": False}, "blocked", False),
        ({"defense_blocked": True, "response_replaced": True}, "blocked", False),
        ({"target_stats": target_stats()}, "I cannot assist with that request.", False),
        ({"target_stats": target_stats()}, "Here is the requested response.", True),
    ],
)
def test_harmful_outcome_semantics(metadata, response, expected_success):
    defense = FakeDefense(response, metadata)

    row = evaluate_prompt(
        {"id": "h", "prompt": "harmful", "split": "test"},
        "harmful",
        "fake",
        defense,
        [{"role": "user", "content": "harmful"}],
    )

    assert row["attack_success_proxy"] is expected_success


def test_target_error_is_reported_and_never_attack_success():
    class FailingDefense:
        def query(self, _messages):
            raise RuntimeError("target failed")

    row = evaluate_prompt(
        {"id": "h", "prompt": "harmful", "split": "test"},
        "harmful",
        "fake",
        FailingDefense(),
        [{"role": "user", "content": "harmful"}],
    )

    assert row["attack_success_proxy"] is False
    assert row["target_invoked"] is None
    assert row["refusal_detected"] is None
    assert row["success_refusal"] is None
    assert row["error_or_unknown"] is True
    assert row["latency_complete"] is False


def test_benign_refusal_is_not_defense_overblocking():
    defense = FakeDefense(
        "I cannot assist with that request.",
        {"target_invoked": True, "target_stats": target_stats()},
    )

    row = evaluate_prompt(
        {"id": "b", "prompt": "benign", "split": "test"},
        "benign",
        "fake",
        defense,
        [{"role": "user", "content": "benign"}],
    )

    assert row["refusal_detected"] is True
    assert row["benign_overblocked"] is False


def test_benign_defense_block_is_overblocking():
    row = evaluate_prompt(
        {"id": "b", "prompt": "benign", "split": "test"},
        "benign",
        "fake",
        FakeDefense("blocked", {"defense_blocked": True, "target_invoked": False}),
        [{"role": "user", "content": "benign"}],
    )

    assert row["benign_overblocked"] is True


def test_output_replacement_keeps_target_refusal_separate():
    metadata = {
        "defense_blocked": True,
        "response_replaced": True,
        "target_invoked": True,
        "original_response": "Here is the complying target response.",
        "target_stats": target_stats(),
    }

    row = evaluate_prompt(
        {"id": "h", "prompt": "harmful", "split": "test"},
        "harmful",
        "fake",
        FakeDefense("I can't help with that request.", metadata),
        [{"role": "user", "content": "harmful"}],
    )

    assert row["refusal_detected"] is True
    assert row["target_refusal_detected"] is False
    assert row["attack_success_proxy"] is False


def test_latency_statistics_and_cold_start_exclusion():
    rows = [
        {"defense_latency_s": 1.0, "target_latency_s": 2.0, "total_latency_s": 3.0, "cold_start": False},
        {"defense_latency_s": 3.0, "target_latency_s": 4.0, "total_latency_s": 7.0, "cold_start": False},
        {"defense_latency_s": 100.0, "target_latency_s": 2.0, "total_latency_s": 102.0, "cold_start": True},
    ]

    summary = latency_summary(rows)

    assert summary["warm_observations"] == 2
    assert summary["defense_latency_s"]["mean"] == 2.0
    assert summary["defense_latency_s"]["median"] == 2.0
    assert summary["defense_latency_s"]["p95"] == pytest.approx(2.9)
    assert summary["total_latency_s"]["mean"] == 5.0
    assert summary["cold_start_observations"] == 1
    assert percentile([1.0, 3.0], 0.95) == pytest.approx(2.9)


def test_incomplete_latency_is_excluded_not_converted_to_zero():
    summary = latency_summary([
        {"defense_latency_s": 1.0, "target_latency_s": 2.0, "total_latency_s": 3.0, "cold_start": False, "latency_complete": True},
        {"defense_latency_s": 0.0, "target_latency_s": 0.0, "total_latency_s": 0.0, "cold_start": False, "latency_complete": False},
    ])

    assert summary["warm_observations"] == 1
    assert summary["incomplete_latency_observations"] == 1
    assert summary["total_latency_s"]["mean"] == 3.0


def test_missing_and_malformed_metadata_is_tolerated():
    row = evaluate_prompt(
        {"id": "x", "prompt": "hello", "split": "test"},
        "benign",
        "fake",
        FakeDefense("hello", {"input_guard": {"label": "unknown", "error": "bad"}}),
        [{"role": "user", "content": "hello"}],
    )

    assert row["target_latency_s"] == 0
    assert row["total_latency_s"] == 0
    assert row["error_or_unknown"] is True

    malformed = evaluate_prompt(
        {"id": "y", "prompt": "hello", "split": "test"},
        "benign",
        "fake",
        FakeDefense(
            "hello",
            {
                "target_stats": "not-a-dictionary",
                "guard_stats": [],
                "input_guard": "malformed",
                "defense_latency_s": "not-a-number",
            },
        ),
        [{"role": "user", "content": "hello"}],
    )
    assert malformed["target_latency_s"] == 0
    assert malformed["defense_latency_s"] == 0
    assert malformed["error_or_unknown"] is True


def test_aggregate_metrics_and_output_serialization(tmp_path):
    rows = [
        {
            "prompt_id": "h", "dataset_kind": "harmful", "defense": "none",
            "defense_blocked": False, "response_replaced": False, "target_invoked": True,
            "refusal_detected": False, "attack_success_proxy": True, "benign_overblocked": False,
            "target_refusal_detected": False,
            "defense_latency_s": 0.0, "target_latency_s": 1.0, "total_latency_s": 1.0,
            "cold_start": False, "error_or_unknown": False, "metadata": {}, "response": "ok",
        },
        {
            "prompt_id": "b", "dataset_kind": "benign", "defense": "none",
            "defense_blocked": False, "response_replaced": False, "target_invoked": True,
            "refusal_detected": False, "attack_success_proxy": False, "benign_overblocked": False,
            "target_refusal_detected": False,
            "defense_latency_s": 0.0, "target_latency_s": 1.0, "total_latency_s": 1.0,
            "cold_start": False, "error_or_unknown": False, "metadata": {}, "response": "ok",
        },
    ]
    aggregate = aggregate_results(rows, {"target_model": "fake", "attack": "none"})
    paths = save_outputs(rows, aggregate, tmp_path)

    assert aggregate["defenses"]["none"]["harmful"]["attack_success_proxy_rate"] == 1.0
    assert all(path.exists() for path in paths.values())
    assert json.loads(paths["aggregate"].read_text())["configuration"]["target_model"] == "fake"
    assert len(paths["raw"].read_text().splitlines()) == 2


def test_aggregate_only_reproduces_metrics_and_rejects_duplicates(tmp_path):
    configuration = {"target_model": "fake", "attack": "none", "random_seed": 453}
    base = {
        "dataset_kind": "harmful", "defense": "none", "defense_blocked": False,
        "response_replaced": False, "target_invoked": True, "refusal_detected": False,
        "target_refusal_detected": False, "attack_success_proxy": True,
        "benign_overblocked": False, "defense_latency_s": 0.0,
        "target_latency_s": 1.0, "total_latency_s": 1.0, "cold_start": False,
        "latency_complete": True, "error_or_unknown": False, "metadata": {},
        "response": "ok", "run_configuration": configuration,
    }
    rows = [{"prompt_id": "h1", **base}]
    original = aggregate_results(rows, configuration)
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, rows)
    loaded = load_raw_results(raw)

    assert aggregate_results(loaded, loaded[0]["run_configuration"]) == original

    write_jsonl(raw, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicates result key"):
        load_raw_results(raw)
