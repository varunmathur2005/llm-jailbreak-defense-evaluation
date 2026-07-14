"""
run.py — Jailbreak attack vs. inference-time defense evaluation harness.

PURPOSE:  Controlled, local, defensive evaluation for a university computer-security
course.  Harmful behaviors are drawn from the JailbreakBench public benchmark dataset
(https://jailbreakbench.github.io/) and are never hardcoded in this repository.
All inference runs against a locally-served Ollama instance; no data leaves the
machine.

IMPORTANT: The results/ directory contains model responses to sensitive prompts.
It is excluded from version control via .gitignore — do not commit it.
"""

import argparse
import csv
import json
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from time import perf_counter

import ollama
from rich.console import Console
from rich.table import Table
from rich import box

import config
from model_client import ModelClient
from attacks import get_attack
from defenses import defense_requires_guard, get_defense
from judge import refusal_check

console = Console()


# ── Startup checks ────────────────────────────────────────────────────────────

def check_ollama(base_url: str, required_models: list[str]) -> None:
    """Verify Ollama is reachable and all required models are pulled."""
    try:
        raw_client = ollama.Client(host=base_url)
        pulled = {m.model for m in raw_client.list().models}
    except Exception:
        console.print(
            f"\n[bold red]ERROR[/bold red]: Cannot connect to Ollama at {base_url}.\n"
            "Make sure Ollama is running.  Start it with:\n\n"
            "  [bold]ollama serve[/bold]\n"
        )
        sys.exit(1)

    missing = [m for m in required_models if m not in pulled]
    if missing:
        console.print("\n[bold red]ERROR[/bold red]: The following models are not pulled:\n")
        for m in missing:
            console.print(f"  [bold]ollama pull {m}[/bold]")
        console.print()
        sys.exit(1)


_Behavior = namedtuple("_Behavior", ["behavior_id", "behavior", "category"])


def load_behaviors(n: int) -> list:
    """Load the first N behaviors from the JailbreakBench dataset (v0.1.0 API).

    jailbreakbench 0.1.0 exposes three parallel lists:
      dataset.behaviors  — short name, e.g. "Defamation"
      dataset.goals      — full harmful request string (the actual prompt text)
      dataset.categories — category string
    We zip them into a lightweight namedtuple so the rest of the code is unchanged.
    """
    try:
        import jailbreakbench as jbb  # noqa: PLC0415
        dataset = jbb.read_dataset()
        records = [
            _Behavior(behavior_id=beh, behavior=goal, category=cat)
            for beh, goal, cat in zip(dataset.behaviors, dataset.goals, dataset.categories)
        ]
        return records[:n]
    except ImportError:
        console.print("[bold red]ERROR[/bold red]: jailbreakbench is not installed.  Run:\n\n"
                      "  pip install jailbreakbench\n")
        sys.exit(1)
    except Exception as exc:
        console.print(
            f"[bold red]ERROR[/bold red]: Failed to load JailbreakBench dataset: {exc}\n\n"
            "The dataset is fetched from the internet on first use and cached locally.\n"
            "Run once with internet access:\n\n"
            "  python -c \"import jailbreakbench as jbb; jbb.read_dataset()\"\n"
        )
        sys.exit(1)


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_condition(
    messages: list[dict],
    defense,
    verbose: bool = False,
) -> dict:
    """Run one (messages, defense) pair and return refusal/latency metrics."""
    if verbose:
        _print_messages(messages)

    start = perf_counter()
    response, meta = defense.query(messages)
    defense_elapsed_s = perf_counter() - start

    guard_label = meta.get("guard_label", "")
    guard_stats = meta.get("guard_stats", {})

    target_stats = meta.get("target_stats", {})
    target_latency_s = target_stats.get("latency_s", 0)
    guard_latency_s = guard_stats.get("latency_s", 0)
    defense_latency_s = meta.get("defense_latency_s", 0)
    active_defense_latency_s = (
        defense_latency_s if "defense_latency_s" in meta else guard_latency_s
    )
    total_latency_s = target_latency_s + active_defense_latency_s
    if not total_latency_s:
        total_latency_s = defense_elapsed_s

    target_total_tokens = target_stats.get("total_tokens", 0)
    guard_total_tokens = guard_stats.get("total_tokens", 0)

    if verbose:
        _print_response(response)

    refusal_detected = refusal_check(response)

    return {
        "response": response,
        "guard_label": guard_label,
        "refusal_detected": refusal_detected,
        # Backward-compatible legacy name: True means attack success/no refusal.
        "success_refusal": not refusal_detected,
        "blocked_by_guard": meta.get("blocked_by_guard", False),
        "response_replaced": meta.get("response_replaced", False),
        "defense_name": meta.get("defense_name", ""),
        "defense_stage": meta.get("defense_stage", ""),
        "defense_blocked": meta.get("defense_blocked", False),
        "defense_latency_s": defense_latency_s,
        "target_invoked": meta.get("target_invoked", True),
        "perplexity_status": meta.get("perplexity_status", ""),
        "perplexity_measured": meta.get("perplexity_measured", False),
        "perplexity_score": meta.get("perplexity_score"),
        "perplexity_threshold": meta.get("perplexity_threshold"),
        "perplexity_token_count": meta.get("perplexity_token_count", 0),
        "perplexity_predicted_token_count": meta.get(
            "perplexity_predicted_token_count", 0
        ),
        "perplexity_model": meta.get("perplexity_model", ""),
        "perplexity_device": meta.get("perplexity_device", ""),
        "perplexity_stride": meta.get("perplexity_stride", 0),
        "perplexity_latency_ms": meta.get("perplexity_latency_ms", 0),
        "perplexity_model_loaded_this_query": meta.get(
            "perplexity_model_loaded_this_query", False
        ),
        "perplexity_model_load_latency_ms": meta.get(
            "perplexity_model_load_latency_ms", 0
        ),
        "perplexity_failure_policy": meta.get("perplexity_failure_policy", ""),
        "perplexity_error": meta.get("perplexity_error", ""),
        "llama_guard_mode": meta.get("llama_guard_mode", ""),
        "llama_guard_model": meta.get("llama_guard_model", ""),
        "llama_guard_failure_policy": meta.get("llama_guard_failure_policy", ""),
        "input_guard": meta.get("input_guard"),
        "output_guard": meta.get("output_guard"),
        "guard_total_latency_s": meta.get("guard_total_latency_s", guard_latency_s),
        "target_latency_s": target_latency_s,
        "guard_latency_s": guard_latency_s,
        "total_latency_s": total_latency_s,
        "target_prompt_tokens": target_stats.get("prompt_tokens", 0),
        "target_completion_tokens": target_stats.get("completion_tokens", 0),
        "target_total_tokens": target_total_tokens,
        "target_tokens_per_second": target_stats.get("tokens_per_second", 0),
        "guard_prompt_tokens": guard_stats.get("prompt_tokens", 0),
        "guard_completion_tokens": guard_stats.get("completion_tokens", 0),
        "guard_total_tokens": guard_total_tokens,
        "guard_tokens_per_second": guard_stats.get("tokens_per_second", 0),
        "total_tokens": target_total_tokens + guard_total_tokens,
    }


def _print_messages(messages: list[dict]) -> None:
    console.print()
    for msg in messages:
        role = msg["role"].upper()
        color = {"system": "yellow", "user": "cyan", "assistant": "green"}.get(msg["role"], "white")
        console.rule(f"[{color}]{role}[/{color}]", style=color)
        console.print(msg["content"])


def _print_response(response: str) -> None:
    console.rule("[green]ASSISTANT[/green]", style="green")
    console.print(response)
    console.rule(style="dim")


def run(
    attack_name: str,
    defense_name: str,
    n: int,
    target_model: str,
    guard_model: str,
    base_url: str,
    verbose: bool = False,
    skip_startup_check: bool = False,
    run_id: str | None = None,
) -> list[dict]:
    """Full evaluation: returns one record per behavior."""
    uses_guard_defense = defense_requires_guard(defense_name)
    if not skip_startup_check:
        required = [target_model]
        if uses_guard_defense:
            required.append(guard_model)
        check_ollama(base_url, required)

    target_client = ModelClient(target_model, base_url)
    guard_client = ModelClient(guard_model, base_url) if uses_guard_defense else None

    attack = get_attack(attack_name)
    with_defense = get_defense(defense_name, target_client, guard_client)

    behaviors = load_behaviors(n)

    console.print(
        f"\n[bold]Jailbreak evaluation[/bold]  "
        f"attack=[cyan]{attack_name}[/cyan]  "
        f"defense=[cyan]{defense_name}[/cyan]  "
        f"model=[cyan]{target_model}[/cyan]  "
        f"N=[cyan]{n}[/cyan]\n"
    )

    records = []
    checkpoint_meta = {
        "attack": attack_name,
        "defense": defense_name,
        "n_behaviors": n,
        "target_models": [target_model],
        "guard_model": guard_model,
        "timestamp": datetime.now().isoformat(),
        "status": "partial",
        "completed_behaviors": 0,
    }
    for idx, b in enumerate(behaviors):
        behavior_text = b.behavior
        behavior_id = getattr(b, "behavior_id", str(idx + 1))

        console.print(
            f"[dim]({idx + 1}/{n})[/dim] [bold]{behavior_id}[/bold]: "
            f"{behavior_text[:80]}{'…' if len(behavior_text) > 80 else ''}"
        )

        messages = attack.build_messages(behavior_text)

        if verbose:
            console.print(
                f"\n[bold yellow]── Selected defense ({defense_name}) ──[/bold yellow]"
            )
        cond = evaluate_condition(messages, with_defense, verbose=verbose)

        _log_condition(defense_name, cond)

        records.append({
            "target_model": target_model,
            "attack": attack_name,
            "defense": defense_name,
            "behavior_id": behavior_id,
            "behavior": behavior_text,
            "prompt_sent": messages[0]["content"],
            "guard_label": cond["guard_label"],
            "refusal_detected": cond["refusal_detected"],
            "success_refusal": cond["success_refusal"],
            "blocked_by_guard": cond["blocked_by_guard"],
            "response_replaced": cond["response_replaced"],
            "defense_name": cond["defense_name"],
            "defense_stage": cond["defense_stage"],
            "defense_blocked": cond["defense_blocked"],
            "defense_latency_s": cond["defense_latency_s"],
            "target_invoked": cond["target_invoked"],
            "perplexity_status": cond["perplexity_status"],
            "perplexity_measured": cond["perplexity_measured"],
            "perplexity_score": cond["perplexity_score"],
            "perplexity_threshold": cond["perplexity_threshold"],
            "perplexity_token_count": cond["perplexity_token_count"],
            "perplexity_predicted_token_count": cond[
                "perplexity_predicted_token_count"
            ],
            "perplexity_model": cond["perplexity_model"],
            "perplexity_device": cond["perplexity_device"],
            "perplexity_stride": cond["perplexity_stride"],
            "perplexity_latency_ms": cond["perplexity_latency_ms"],
            "perplexity_model_loaded_this_query": cond[
                "perplexity_model_loaded_this_query"
            ],
            "perplexity_model_load_latency_ms": cond[
                "perplexity_model_load_latency_ms"
            ],
            "perplexity_failure_policy": cond["perplexity_failure_policy"],
            "perplexity_error": cond["perplexity_error"],
            "llama_guard_mode": cond["llama_guard_mode"],
            "llama_guard_model": cond["llama_guard_model"],
            "llama_guard_failure_policy": cond["llama_guard_failure_policy"],
            "input_guard": cond["input_guard"],
            "output_guard": cond["output_guard"],
            "guard_total_latency_s": cond["guard_total_latency_s"],
            "target_latency_s": cond["target_latency_s"],
            "guard_latency_s": cond["guard_latency_s"],
            "total_latency_s": cond["total_latency_s"],
            "target_prompt_tokens": cond["target_prompt_tokens"],
            "target_completion_tokens": cond["target_completion_tokens"],
            "target_total_tokens": cond["target_total_tokens"],
            "target_tokens_per_second": cond["target_tokens_per_second"],
            "guard_prompt_tokens": cond["guard_prompt_tokens"],
            "guard_completion_tokens": cond["guard_completion_tokens"],
            "guard_total_tokens": cond["guard_total_tokens"],
            "guard_tokens_per_second": cond["guard_tokens_per_second"],
            "total_tokens": cond["total_tokens"],
            "response": cond["response"],
        })
        checkpoint_meta["completed_behaviors"] = len(records)
        save_checkpoint(records, checkpoint_meta, run_id)

    return records


def run_suite(
    attack_name: str,
    defense_name: str,
    n: int,
    target_models: list[str],
    guard_model: str,
    base_url: str,
    verbose: bool = False,
    run_id: str | None = None,
) -> list[dict]:
    """Evaluate the same attack/defense across multiple target models."""
    required = [*target_models]
    if defense_requires_guard(defense_name):
        required.append(guard_model)
    check_ollama(base_url, required)

    all_records = []
    for idx, target_model in enumerate(target_models):
        if idx:
            console.rule(style="dim")

        records = run(
            attack_name=attack_name,
            defense_name=defense_name,
            n=n,
            target_model=target_model,
            guard_model=guard_model,
            base_url=base_url,
            verbose=verbose,
            skip_startup_check=True,
            run_id=run_id,
        )
        print_summary_table(records, defense_name, title_suffix=f" — {target_model}")
        all_records.extend(records)

    print_model_comparison_table(all_records, defense_name)
    return all_records


def _log_condition(label: str, cond: dict) -> None:
    refused = not cond["success_refusal"]
    refused_str = "[green]YES[/green]" if refused else "[red]NO[/red]"
    extra = ""
    if cond["guard_label"]:
        blocked_str = "[green]YES[/green]" if cond["blocked_by_guard"] else "[yellow]NO[/yellow]"
        extra = f"  guard_blocked={blocked_str}"
    console.print(
        f"    {label:<20} refused={refused_str}  "
        f"latency={cond['total_latency_s']:.2f}s  tokens={cond['total_tokens']}{extra}"
    )


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary_table(records: list[dict], defense_name: str, title_suffix: str = "") -> None:
    """Print a Rich before/after summary table."""
    table = Table(
        title=f"Jailbreak Attack vs. Defense — Summary{title_suffix}",
        box=box.ROUNDED,
        show_footer=True,
    )

    n = len(records)

    # Column helpers
    def asr(values: list[bool]) -> str:
        rate = sum(values) / n if n else 0
        return f"{rate:.0%} ASR"

    def avg_s(values: list[float]) -> str:
        return f"{sum(values) / n:.2f}s" if n else "0.00s"

    success = [r["success_refusal"] for r in records]
    latencies = [r["total_latency_s"] for r in records]

    table.add_column("Behavior ID", footer="AGGREGATE", style="bold", min_width=16)
    table.add_column("No-Refusal", footer=asr(success), justify="center", min_width=14)
    table.add_column("Latency", footer=avg_s(latencies), justify="center", min_width=12)

    for r in records:
        table.add_row(
            r["behavior_id"],
            _bool_cell(r["success_refusal"]),
            f"{r['total_latency_s']:.2f}s",
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Columns: No-Refusal = model did not refuse (attack success by keyword check); "
        "Latency = target model time plus any active defense time.[/dim]\n"
    )


def _bool_cell(success: bool) -> str:
    return "[red]✓ success[/red]" if success else "[green]✗ blocked[/green]"


def print_model_comparison_table(records: list[dict], defense_name: str) -> None:
    """Print one aggregate row per target model for suite runs."""
    grouped = {}
    for record in records:
        grouped.setdefault(record["target_model"], []).append(record)

    table = Table(
        title="Model Suite Comparison",
        box=box.ROUNDED,
    )
    table.add_column("Target model", style="bold", min_width=20)
    table.add_column("No-Refusal", justify="center")
    table.add_column("Avg latency", justify="center")
    table.add_column("Avg tokens", justify="center")

    def asr(rows: list[dict], key: str) -> str:
        return f"{sum(r[key] for r in rows) / len(rows):.0%} ASR" if rows else "n/a"

    def avg(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0

    for model, rows in grouped.items():
        table.add_row(
            model,
            asr(rows, "success_refusal"),
            f"{avg(rows, 'total_latency_s'):.2f}s",
            f"{avg(rows, 'total_tokens'):.0f}",
        )

    console.print()
    console.print(table)
    console.print()


def save_results(records: list[dict], run_meta: dict) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown result files."""
    out_dir = Path(config.RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"run_{ts}.json"
    csv_path = out_dir / f"run_{ts}.csv"
    md_path = out_dir / f"run_{ts}.md"

    payload = {"meta": run_meta, "results": records}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    md_path.write_text(build_markdown_report(records, run_meta, json_path, csv_path), encoding="utf-8")

    return json_path, csv_path, md_path


def save_checkpoint(
    records: list[dict],
    run_meta: dict,
    run_id: str | None = None,
) -> tuple[Path, Path, Path]:
    """Overwrite a recoverable partial result after each completed behavior."""
    out_dir = Path(config.RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _safe_slug("_".join(run_meta["target_models"]))
    attack_slug = _safe_slug(run_meta["attack"])
    defense_slug = _safe_slug(run_meta["defense"])
    stem = f"checkpoint_{run_id}_{model_slug}_{attack_slug}_{defense_slug}"

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    payload = {"meta": run_meta, "results": records}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    md_path.write_text(build_markdown_report(records, run_meta, json_path, csv_path), encoding="utf-8")
    return json_path, csv_path, md_path


def build_markdown_report(
    records: list[dict],
    run_meta: dict,
    json_path: Path,
    csv_path: Path,
) -> str:
    """Create a presentation-friendly report without embedding harmful responses."""
    lines = [
        "# Jailbreak Defense POC Report",
        "",
        "## Run",
        "",
        f"- Attack: `{run_meta['attack']}`",
        f"- Defense: `{run_meta['defense']}`",
        f"- Behaviors: `{run_meta['n_behaviors']}`",
        f"- Target models: `{', '.join(run_meta['target_models'])}`",
        f"- Guard model: `{run_meta['guard_model']}`" if defense_requires_guard(run_meta["defense"]) else "- Guard model: `not used`",
        f"- Timestamp: `{run_meta['timestamp']}`",
        f"- Raw JSON: `{json_path.name}`",
        f"- Raw CSV: `{csv_path.name}`",
        "",
        "## Results By Model",
        "",
    ]

    for model, rows in _group_by_model(records).items():
        stats = _summarize(rows)
        lines.extend([
            f"### {model}",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| No-refusal ASR | {_fmt_pct(stats['refusal_asr'])} |",
            f"| Avg total latency | {stats['latency']:.2f}s |",
            f"| Avg target latency | {stats['target_latency']:.2f}s |",
            f"| Avg total tokens | {stats['tokens']:.0f} |",
            f"| Avg target tokens/sec | {stats['tps']:.0f} |",
            "",
            "| Behavior | No-refusal | Latency |",
            "|---|---:|---:|",
        ])

        for row in rows:
            lines.append(
                "| "
                f"{_escape_md(row['behavior_id'])} | "
                f"{_yes_no(row['success_refusal'])} | "
                f"{row['total_latency_s']:.2f}s |"
            )
        lines.append("")

    overall = _summarize(records)
    lines.extend([
        "## Overall Statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total evaluated rows | {len(records)} |",
        f"| No-refusal ASR | {_fmt_pct(overall['refusal_asr'])} |",
        f"| Avg total latency | {overall['latency']:.2f}s |",
        f"| Avg target latency | {overall['target_latency']:.2f}s |",
        f"| Avg total tokens | {overall['tokens']:.0f} |",
        f"| Avg target tokens/sec | {overall['tps']:.0f} |",
        "",
        "## Notes",
        "",
        "- No-refusal ASR means the model did not use a refusal phrase, so lower is better.",
        "- This report intentionally omits full prompts and model responses. Use the raw JSON/CSV for audit details.",
        "",
    ])
    return "\n".join(lines)


def _group_by_model(records: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for record in records:
        grouped.setdefault(record["target_model"], []).append(record)
    return grouped


def _summarize(rows: list[dict]) -> dict[str, float]:
    def rate(key: str) -> float:
        return sum(1 for row in rows if row[key]) / len(rows) if rows else 0

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0

    return {
        "refusal_asr": rate("success_refusal"),
        "latency": avg("total_latency_s"),
        "target_latency": avg("target_latency_s"),
        "tokens": avg("total_tokens"),
        "tps": avg("target_tokens_per_second"),
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.0%}"


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def _escape_md(value: object) -> str:
    return str(value).replace("|", "\\|")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate jailbreak attacks vs. inference-time defenses.",
    )
    parser.add_argument("--attack", default=config.ATTACK,
                        help=f"Attack to use (default: {config.ATTACK})")
    parser.add_argument("--defense", default=config.DEFENSE,
                        help=f"Defense to use (default: {config.DEFENSE})")
    parser.add_argument("--n", type=int, default=config.N_BEHAVIORS,
                        help=f"Number of behaviors (default: {config.N_BEHAVIORS})")
    parser.add_argument("--model", default=None,
                        help="Run one target model instead of the configured suite")
    parser.add_argument("--models", default=",".join(config.TARGET_MODELS),
                        help="Comma-separated target model suite "
                             f"(default: {', '.join(config.TARGET_MODELS)})")
    parser.add_argument("--guard-model", default=config.GUARD_MODEL,
                        help=f"Guard model (default: {config.GUARD_MODEL})")
    parser.add_argument("--ollama-url", default=config.OLLAMA_BASE_URL,
                        help=f"Ollama base URL (default: {config.OLLAMA_BASE_URL})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every prompt and response as it runs")
    return parser.parse_args()


def _parse_models(raw: str) -> list[str]:
    """Parse a comma-separated model list while preserving order."""
    models = []
    seen = set()
    for item in raw.split(","):
        model = item.strip()
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    return models


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_models = [args.model] if args.model else _parse_models(args.models)
    if not target_models:
        console.print("[bold red]ERROR[/bold red]: At least one target model is required.")
        sys.exit(1)

    try:
        if len(target_models) == 1:
            records = run(
                attack_name=args.attack,
                defense_name=args.defense,
                n=args.n,
                target_model=target_models[0],
                guard_model=args.guard_model,
                base_url=args.ollama_url,
                verbose=args.verbose,
                run_id=run_id,
            )
            print_summary_table(records, args.defense)
        else:
            records = run_suite(
                attack_name=args.attack,
                defense_name=args.defense,
                n=args.n,
                target_models=target_models,
                guard_model=args.guard_model,
                base_url=args.ollama_url,
                verbose=args.verbose,
                run_id=run_id,
            )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted. Check results/checkpoint_* for partial JSON, CSV, and Markdown files.[/yellow]\n"
        )
        sys.exit(130)

    run_meta = {
        "attack": args.attack,
        "defense": args.defense,
        "n_behaviors": args.n,
        "target_models": target_models,
        "guard_model": args.guard_model,
        "timestamp": datetime.now().isoformat(),
    }
    json_path, csv_path, md_path = save_results(records, run_meta)
    console.print(f"[dim]Results saved to:[/dim]\n  {json_path}\n  {csv_path}\n  {md_path}\n")


if __name__ == "__main__":
    main()
