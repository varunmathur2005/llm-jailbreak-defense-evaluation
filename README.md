# Jailbreak PoC — Inference-Time Defense Evaluation

A minimal, self-contained proof-of-concept for a university computer-security course.
It runs harmful-behavior prompts from JailbreakBench through a target language
model under one selected attack and defense condition. Baseline and defended
conditions are run separately and compared through the local reporting utility.

All inference runs locally via [Ollama](https://ollama.com) — no data leaves the
machine after the one-time model pull.

---

## Requirements

| Requirement | Notes |
|---|---|
| Apple Silicon Mac (M1/M2/M3) | Metal-accelerated Ollama inference |
| macOS 13+ | |
| Python 3.11+ | |
| [Ollama](https://ollama.com) | Free, local LLM server |
| Internet (one-time) | Pull models + JailbreakBench dataset cache |

---

## Setup

### 1. Install Ollama

```bash
brew install ollama
# or download from https://ollama.com
```

Start Ollama in another terminal with `ollama serve`, or use the Ollama app.

### 2. Pull the required models

```bash
ollama pull dolphin-mistral:7b # weaker / less-refusal target baseline
ollama pull qwen2.5:3b         # medium small target model
ollama pull llama3.2:3b        # stronger default-safe target model
ollama pull llama-guard3:1b    # guard/classifier model
```

For a 16 GB machine, you can add another less-restrictive target:

```bash
ollama pull dolphin-llama3:8b
```

### 3. Create a virtual environment and install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Cache the JailbreakBench dataset (one-time, requires internet)

```bash
python -c "import jailbreakbench as jbb; jbb.read_dataset()"
```

After this step the dataset is cached locally and the tool runs fully offline.

---

## Usage

```bash
# Make sure Ollama is running
ollama serve &   # or launch the Ollama app

# Default run (template attack vs. llama_guard defense, configured model suite)
python run.py

# Override the defense or sample size from the command line
python run.py --attack template --defense self_reminder --n 10

# Debug one target model instead of the full suite
python run.py --model qwen2.5:3b --n 5

# Run a custom target suite
python run.py --models dolphin-mistral:7b,qwen2.5:3b,llama3.2:3b --n 25

# Baseline: raw behavior, no defense
python run.py --attack none --defense none --n 5
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--attack` | `template` | `none` or `template` |
| `--defense` | `llama_guard` | `none`, `self_reminder`, `llama_guard_input`, `llama_guard`, `llama_guard_both`, `both`, or `perplexity` |
| `--n` | `25` | Number of JBB behaviors to evaluate (max 100) |
| `--model` | unset | Run one Ollama target model instead of the configured suite |
| `--models` | `dolphin-mistral:7b,qwen2.5:3b,llama3.2:3b` | Comma-separated Ollama target model suite |
| `--guard-model` | `llama-guard3:1b` | Ollama Llama Guard model tag |
| `--ollama-url` | `http://localhost:11434` | Ollama API base URL |

All defaults live in `config.py`.

---

## How it works

```text
JailbreakBench → attack.build_messages() → selected defense.query()
                                              ├─ input checks
                                              ├─ Ollama target
                                              └─ output checks
                                                        ↓
                                refusal proxy + JSON/CSV/Markdown results
```

Each `run.py` invocation evaluates one selected defense. Run `--defense none`
as the baseline, or use `scripts.compare_varun_defenses` to execute matched
baseline and defended conditions. Llama Guard is a defense, not the ASR judge.

The default model suite compares defenses across targets with different baseline
refusal behavior: a weaker/less-refusal model, a medium small model, and a
stronger default-safe instruction model. This makes it easier to explain when
inference-time defenses add value and when the base model already refuses.

### Attacks

| Name | Description |
|---|---|
| `none` | Raw behavior text, no wrapper — baseline |
| `template` | DAN-style "Developer Mode" wrapper (refusal-suppression template from academic literature; harmful behaviors come from JBB, not this repo) |

### Defenses

| Name | Type | Description |
|---|---|---|
| `none` | — | No defense — measures raw model compliance |
| `self_reminder` | Pre-hoc | Adds a safety-reinforcing system prompt before the user turn |
| `llama_guard_input` | Input filter | Classifies the user conversation before target inference |
| `llama_guard` | Output filter | Classifies and optionally replaces the target response |
| `llama_guard_both` | Input/output filter | Checks input, then checks output if input is safe |
| `perplexity` | Input filter | Blocks unusually high-perplexity user prompts before target inference |

### Varun defenses and local evaluation

#### Perplexity

`perplexity` scores only the latest user message with local GPT-2 and blocks
when `score > PERPLEXITY_THRESHOLD`; equality is allowed. Blocked prompts do not
invoke the target. Model, threshold, device, stride, failure policy, and blocked
response are configured in `config.py`. The model loads lazily and is cached by
Hugging Face after its first use.

Calibrate the threshold on the dedicated calibration split before experiments:

```bash
python -m scripts.calibrate_perplexity
```

The calibration cache is resumable and the report distinguishes requested from
achieved benign false-positive rate. With 20 benign calibration prompts, rates
occur in 5-point increments: the 1% budget resolves to 0%, while 5% and 10%
correspond to one and two false positives.

#### Llama Guard

Llama Guard uses the configured Ollama `GUARD_MODEL` through its native chat
interface. `llama_guard_input` blocks before generation, `llama_guard` preserves
the backward-compatible output-only mode, and `llama_guard_both` checks both in
order. Registered variants override `LLAMA_GUARD_MODE`. The configured failure
policy is `allow`, `block`, or `raise`; raw labels, `S1`–`S13` categories, errors,
tokens, latency, and per-stage throughput are preserved.

#### Refusal proxy and comparison metrics

`refusal_detected=true` means the delivered response opened with a recognized
first-person refusal. The legacy `success_refusal` field is its inverse and means
attack success by the no-refusal proxy. The comparison utility additionally
requires that the defense neither blocked nor replaced the delivered response.
Llama Guard is not reused as the ASR judge.

For benign prompts, over-blocking means an input block or output replacement;
target refusal is reported separately. Total latency is target plus defense
latency once. Identified GPT-2 cold loads and incomplete latency observations are
reported separately from warm mean, median, and linearly interpolated p95.

#### Commands

```bash
# Unit tests (no model downloads or Ollama calls)
python -m pytest -q

# One-prompt smoke tests
python run.py --model qwen2.5:3b --attack none --defense perplexity --n 1
python run.py --model qwen2.5:3b --attack none --defense llama_guard_input --n 1
python run.py --model qwen2.5:3b --attack none --defense llama_guard --n 1
python run.py --model qwen2.5:3b --attack none --defense llama_guard_both --n 1

# Small matched comparison; replace 100.0 with the calibrated threshold
python -m scripts.compare_varun_defenses \
  --target-model qwen2.5:3b --attack none \
  --perplexity-threshold 100.0 \
  --n-harmful 2 --n-benign 2 --include-both

# Full local comparison
python -m scripts.compare_varun_defenses \
  --target-model qwen2.5:3b --attack template \
  --perplexity-threshold 100.0 \
  --n-harmful 100 --n-benign 20 --include-both

# Regenerate reports without models or network access
python -m scripts.compare_varun_defenses \
  --aggregate-only results/varun_<timestamp>/raw_results.jsonl \
  --output-dir results/varun_<timestamp>_report
```

The comparison uses recorded Ollama warm-up calls and deterministic target
settings (`--temperature 0`, `--seed 453`) by default.

#### Limitations

Plain perplexity can miss fluent jailbreaks and over-block unusual benign text.
Llama Guard can misclassify or be attacked. The refusal proxy can miss nuanced
partial compliance. The included calibration and benign-test fixtures are useful
for implementation checks and preliminary measurements, not broad final claims.

### Judge metrics

| Metric | Meaning |
|---|---|
| **No-refusal proxy** | Delivered response contained no recognized refusal and was not blocked/replaced |
| **Latency** | Target-model time plus active defense calls such as perplexity or Guard |
| **Tokens** | Ollama-reported prompt + completion tokens, useful as a local compute-cost proxy |

Lower attack success and benign over-blocking are better; latency exposes the
security-performance trade-off.

Because this project runs locally through Ollama, there is no per-call API bill.
For cost/latency comparisons, use average total tokens and average latency:

```bash
# Quick latency/cost debug on one model
python run.py --model qwen2.5:3b --defense llama_guard --n 5

# Compare defense overhead across the default model suite
python run.py --defense self_reminder --n 25
python run.py --defense llama_guard --n 25
```

JSON/CSV rows preserve defense decisions, `refusal_detected`, the legacy
`success_refusal`, target/defense/total latency, token counts, and defense-specific
telemetry.

---

## Output

Each run writes to `results/` (git-ignored):

- `results/run_<timestamp>.json` — full records including response snippets
- `results/run_<timestamp>.csv` — flat table, import into Excel / pandas
- `results/run_<timestamp>.md` — concise report without full sensitive responses
- `results/varun_<timestamp>/` — raw and aggregate local comparison reports

---

## Project layout

```
jailbreak-poc/
├── config.py              # all experiment knobs
├── model_client.py        # thin Ollama wrapper
├── attacks/
│   ├── base.py            # BaseAttack ABC
│   ├── none.py            # raw behavior, no template
│   ├── template.py        # DAN-style template attack
│   └── templates/
│       └── dan_template.txt
├── defenses/
│   ├── base.py            # BaseDefense ABC
│   ├── none.py            # passthrough
│   ├── self_reminder.py   # safety system-prompt injection
│   ├── llama_guard.py     # Llama Guard 3 classifier + defense
│   └── perplexity.py      # local perplexity input filter
├── data/
│   ├── perplexity_calibration.jsonl
│   └── benign_test.jsonl
├── scripts/
│   ├── calibrate_perplexity.py
│   └── compare_varun_defenses.py
├── tests/                 # deterministic unit tests; no live model calls
├── judge.py               # refusal_check and success signal helpers
├── run.py                 # main evaluation harness
├── requirements.txt
├── README.md
└── results/               # git-ignored; created at runtime
```

---

## Extending

**Add an attack**: subclass `attacks/base.py:BaseAttack`, implement `build_messages`,
register it in `attacks/__init__.py:REGISTRY`.

**Add a defense**: subclass `defenses/base.py:BaseDefense`, implement `query`,
register it in `defenses/__init__.py:REGISTRY`.

---

## Academic context & ethics

- Behaviors come from [JailbreakBench](https://jailbreakbench.github.io/), a published
  academic benchmark — they are not authored by this project.
- All inference is local.  No harmful content is sent to any external service.
- The goal is to *measure* defense effectiveness, not to enable harm.
- `results/` is `.gitignore`-d to avoid committing model outputs.
