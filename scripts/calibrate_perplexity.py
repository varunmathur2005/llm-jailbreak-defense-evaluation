"""Calibrate a plain perplexity threshold on labelled, held-out prompts."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import config
from defenses.perplexity import HuggingFacePerplexityScorer


TARGET_FALSE_POSITIVE_RATES = (0.01, 0.05, 0.10)
CACHE_VERSION = 1


def load_labelled_prompts(path: Path, required_split: str = "calibration") -> list[dict]:
    records = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} contains malformed JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        if record.get("split") != required_split:
            raise ValueError(
                f"{path}:{line_number} must use split={required_split!r}; "
                "final-test data cannot be used for calibration"
            )
        if record.get("label") not in {"benign", "attack"}:
            raise ValueError(f"{path}:{line_number} has an invalid label")
        if not isinstance(record.get("id"), str) or not isinstance(record.get("prompt"), str):
            raise ValueError(f"{path}:{line_number} requires string id and prompt")
        if record["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number} has duplicate id {record['id']!r}")
        seen_ids.add(record["id"])
        records.append(record)
    if not records or not {r["label"] for r in records} >= {"benign", "attack"}:
        raise ValueError("calibration data must contain benign and attack prompts")
    return records


def score_cache_key(record: dict, model: str, stride: int, device: str) -> str:
    digest = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
    return (
        f"v{CACHE_VERSION}:{model}:{stride}:{device}:"
        f"{record['label']}:{record['id']}:{digest}"
    )


def score_prompts(records: list[dict], scorer, cache_path: Path) -> tuple[list[dict], int]:
    cache = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cache {cache_path} contains malformed JSON") from exc
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            raise ValueError(f"Cache {cache_path} has an unsupported format/version")
        cache = payload.get("scores")
        if not isinstance(cache, dict):
            raise ValueError(f"Cache {cache_path} scores must be an object")

    scores = []
    computed = 0
    for record in records:
        cache_device = _resolved_cache_device(scorer)
        key = score_cache_key(record, scorer.model_name, scorer.stride, cache_device)
        cached = cache.get(key)
        if cached is not None and _valid_cached_score(cached, record, scorer.model_name):
            scores.append(cached)
            continue
        try:
            result = scorer.score(record["prompt"])
            scored = {
                "id": record["id"],
                "label": record["label"],
                "split": record["split"],
                "score": result.score,
                "token_count": result.token_count,
                "predicted_token_count": result.predicted_token_count,
                "model": scorer.model_name,
                "device": result.device,
                "scoring_status": "scored" if result.score is not None else "unscorable",
                "scoring_error": "",
            }
        except Exception as exc:
            scored = {
                "id": record["id"],
                "label": record["label"],
                "split": record["split"],
                "score": None,
                "token_count": 0,
                "predicted_token_count": 0,
                "model": scorer.model_name,
                "device": getattr(scorer, "requested_device", "unknown"),
                "scoring_status": "error",
                "scoring_error": f"{type(exc).__name__}: {exc}",
            }
        cache[key] = scored
        scores.append(scored)
        computed += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"version": CACHE_VERSION, "scores": cache}, indent=2),
            encoding="utf-8",
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"version": CACHE_VERSION, "scores": cache}, indent=2),
        encoding="utf-8",
    )
    return scores, computed


def _resolved_cache_device(scorer) -> str:
    requested = getattr(scorer, "requested_device", "unknown")
    if requested != "auto":
        return requested
    try:
        import torch

        return scorer._resolve_device(torch, "auto")
    except (AttributeError, ImportError):
        return "auto-unknown"


def _valid_cached_score(cached: object, record: dict, model: str) -> bool:
    if not isinstance(cached, dict):
        raise ValueError(f"Corrupt cache row for prompt {record['id']!r}")
    required = {"id", "label", "split", "score", "scoring_status", "scoring_error"}
    if not required <= cached.keys():
        raise ValueError(f"Incomplete cache row for prompt {record['id']!r}")
    if cached["id"] != record["id"] or cached["label"] != record["label"]:
        raise ValueError(f"Mismatched cache row for prompt {record['id']!r}")
    if cached.get("model") != model:
        return False
    status = cached["scoring_status"]
    score = cached.get("score")
    if status == "scored" and not isinstance(score, (int, float)):
        raise ValueError(f"Corrupt measured score for prompt {record['id']!r}")
    if status == "unscorable" and score is not None:
        raise ValueError(f"Corrupt unscorable row for prompt {record['id']!r}")
    # Provider errors can be transient; deliberately retry them on resume.
    return status in {"scored", "unscorable"}


def threshold_metrics(scores: list[dict], threshold: float) -> dict:
    valid = [row for row in scores if row.get("score") is not None]
    benign = [row for row in valid if row["label"] == "benign"]
    attacks = [row for row in valid if row["label"] == "attack"]
    false_positives = sum(row["score"] > threshold for row in benign)
    true_negatives = len(benign) - false_positives
    true_positives = sum(row["score"] > threshold for row in attacks)
    false_negatives = len(attacks) - true_positives
    return {
        "threshold": threshold,
        "benign_false_positive_rate": false_positives / len(benign) if benign else None,
        "attack_detection_rate": true_positives / len(attacks) if attacks else None,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "benign_measured_count": len(benign),
        "attack_measured_count": len(attacks),
    }


def build_threshold_sweep(scores: list[dict]) -> list[dict]:
    values = sorted({float(row["score"]) for row in scores if row.get("score") is not None})
    if not values:
        raise ValueError("no valid perplexity scores are available")
    epsilon = max(abs(values[0]) * 1e-12, 1e-12)
    thresholds = [values[0] - epsilon, *values]
    return [threshold_metrics(scores, threshold) for threshold in thresholds]


def select_thresholds(sweep: list[dict]) -> dict:
    selected = {}
    for target in TARGET_FALSE_POSITIVE_RATES:
        usable = [
            row
            for row in sweep
            if row["benign_false_positive_rate"] is not None
            and row["attack_detection_rate"] is not None
            and row["benign_false_positive_rate"] <= target
        ]
        if not usable:
            selected[f"{int(target * 100)}pct"] = {
                "requested_false_positive_rate": target,
                "selection_status": "no_threshold_within_budget",
            }
            continue
        best = min(
            usable,
            key=lambda row: (
                -row["attack_detection_rate"],
                target - row["benign_false_positive_rate"],
                row["threshold"],
            ),
        )
        selected[f"{int(target * 100)}pct"] = {
            "requested_false_positive_rate": target,
            "achieved_false_positive_rate": best["benign_false_positive_rate"],
            "absolute_rate_difference": abs(best["benign_false_positive_rate"] - target),
            "selection_status": "selected_within_budget",
            **best,
        }
    selected["recommended"] = {
        "basis": "5% benign false-positive target",
        **selected["5pct"],
    }
    return selected


def calibrate(input_path: Path, output_dir: Path, scorer, cache_path: Path) -> dict:
    records = load_labelled_prompts(input_path)
    scores, computed = score_prompts(records, scorer, cache_path)
    sweep = build_threshold_sweep(scores)
    selected = select_thresholds(sweep)
    benign_count = sum(row["label"] == "benign" for row in scores)
    payload = {
        "configuration": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path),
            "required_split": "calibration",
            "model": scorer.model_name,
            "requested_device": getattr(scorer, "requested_device", "unknown"),
            "stride": scorer.stride,
            "cache_file": str(cache_path),
            "new_scores_computed": computed,
            "benign_count": benign_count,
            "attack_count": sum(row["label"] == "attack" for row in scores),
            "small_benign_sample": benign_count < 20,
        },
        "raw_scores": scores,
        "threshold_sweep": sweep,
        "selected_thresholds": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/perplexity_calibration.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/perplexity_calibration"))
    parser.add_argument("--cache", type=Path, default=Path("results/perplexity_calibration/cache.json"))
    parser.add_argument("--model", default=config.PERPLEXITY_MODEL)
    parser.add_argument("--device", default=config.PERPLEXITY_DEVICE)
    parser.add_argument("--stride", type=int, default=config.PERPLEXITY_STRIDE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = HuggingFacePerplexityScorer(args.model, args.device, args.stride)
    payload = calibrate(args.input, args.output_dir, scorer, args.cache)
    recommended = payload["selected_thresholds"]["recommended"]
    if recommended["selection_status"] == "selected_within_budget":
        print(f"Recommended threshold: {recommended['threshold']:.6g}")
        print(
            "Requested/achieved benign FPR: "
            f"{recommended['requested_false_positive_rate']:.1%}/"
            f"{recommended['achieved_false_positive_rate']:.1%}"
        )
    else:
        print("No threshold could be selected within the requested FPR budget.")
    print(f"Calibration saved to: {args.output_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
