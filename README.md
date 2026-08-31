# LLM Jailbreak Defense Evaluation

A modular framework for evaluating **inference-time defenses against LLM jailbreak attacks** across security, usability, and computational cost.

**[Paper](./paper.pdf)**

Built as a University of Waterloo computer security research project by **Ryan Maxin, Vishesh Gupta, Varun Mathur, and Adam Kaegi**.

## Highlights

- Built a composable **Python + LangChain evaluation harness** with interchangeable attacks, defenses, judges, and local LLMs.
- Evaluated **2,500 model responses** across **4 attack conditions and 10 defense conditions** using JailbreakBench.
- Measured attack success with **HarmBench** and refusal / benign over-blocking with **WildGuard**.
- Compared security gains against **latency, target-model calls, and benign usability** rather than optimizing attack success alone.
- Found a three-defense stack reaching **1.5% cross-attack ASR with 80% fewer response-path model calls** than the 1.0% ASR full stack.

---

## Architecture

```text
Prompt
  │
  ▼
Attack
  │
  ▼
Input-stage defenses
  │
  ▼
Target LLM
  │
  ▼
Output-stage defenses
  │
  ▼
Judge
  │
  ▼
Evaluation + reporting
```

Attacks, defenses, and judges implement shared interfaces and are registered independently. The pipeline automatically places each defense at its declared stage, allowing components to be swapped or stacked without modifying the core execution logic.

```text
attacks/     pluggable jailbreak attacks
defenses/    input, generation, and output-stage defenses
judges/      interchangeable safety and refusal evaluators
prompts/     JailbreakBench prompt batches
core/        pipeline, configuration, caching, matrix execution, reporting
scripts/     experiment, model-prefetch, and evaluation utilities
tests/       deterministic tests for pipeline and experiment behavior
main.py      CLI entry point
```

---

## Experiment

The final evaluation used a locally served **Qwen2.5 7B** target model and four harmful attack conditions:

| Attack | Type |
|---|---|
| None | Raw harmful request baseline |
| DeepInception | Nested role-play attack |
| GCG | Transfer-based adversarial suffix attack |
| PAIR | Adaptive black-box prompt refinement |

We evaluated four standalone defenses:

| Defense | Stage |
|---|---|
| Self-Reminder | Input |
| Perplexity | Input |
| SmoothLLM | Generation |
| Llama Guard | Output |

Five additional defense stacks tested whether complementary mechanisms improved robustness enough to justify their added cost.

The resulting matrix contained:

- **50 harmful JailbreakBench prompts**
- **50 benign JailbreakBench prompts**
- **4 harmful attack conditions**
- **10 defense conditions**
- **50 experiment cells**
- **2,500 evaluated responses**

---

## Key Results

| Defense | Cross-Attack ASR | Benign Refusal |
|---|---:|---:|
| No defense | 24.0% | 12% |
| Llama Guard output | 4.5% | 32% |
| Perplexity + Self-Reminder + Llama Guard | **1.5%** | 38% |
| Full stack | **1.0%** | 44% |

The full stack achieved the lowest observed attack success rate, but the improvement over the leaner three-defense stack was only **0.5 percentage points**.

The computational difference was substantially larger:

```text
Three-defense stack: 188 response-path target-model calls
Full stack:          940 response-path target-model calls
```

The three-defense stack therefore used **80% fewer model calls** while remaining within one additional successful attack across 200 harmful evaluations.

The results demonstrate that maximizing the number of defenses does not necessarily produce the best deployment trade-off: **security, benign usability, and inference cost need to be evaluated together.**

---

## Evaluation Metrics

### Attack Success Rate

**HarmBench Mistral** determines whether a generated response actually fulfills the harmful behavior requested by the benchmark.

This is kept separate from refusal detection because a response can avoid explicitly refusing while still failing to provide the requested harmful behavior.

### Refusal and Over-Blocking

**WildGuard** measures:

- refusal on harmful prompts
- false refusal on benign prompts

This captures the usability cost introduced by increasingly aggressive defenses.

### Performance

The framework records:

- response latency
- early input blocking
- target-model calls
- defense-specific latency
- attack construction cost
- refusal and behavior classifications

Adaptive attack construction cost is tracked separately from defended response latency.

---

## Components

### Attacks

- **DeepInception** — nested fictional-scenario attack.
- **GCG** — transfer-based adversarial suffix candidates with black-box reranking.
- **PAIR** — bounded multi-stream black-box prompt refinement.
- **None** — raw-prompt baseline.

### Defenses

- **Self-Reminder** — reinforces safety constraints through prompt augmentation.
- **Perplexity** — blocks statistically anomalous inputs before target inference.
- **SmoothLLM** — evaluates perturbed prompt variants and selects from the majority response class.
- **Llama Guard** — performs semantic safety classification at the input or output stage.

### Judges

- **HarmBench Mistral** — harmful-behavior success classifier.
- **WildGuard** — refusal classifier.
- **StrongREJECT** — continuous harmful-assistance evaluator.
- Additional lightweight judges are included for pipeline testing and debugging.

---

## Reproducibility

A trial is defined by:

```text
target model × attack × defense stack × prompt batch
```

Experiment configurations are data rather than hard-coded execution paths, allowing the full matrix to be generated from the same pipeline.

The framework includes:

- deterministic model settings
- prompt and configuration hashes
- attack caching across defense conditions
- atomic per-cell checkpoints
- resumable experiments
- resumable evaluation
- run manifests
- latency and model-call telemetry

Interrupted runs can therefore continue without regenerating completed model responses.

---

## Setup

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements-cuda.txt
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Fetch required assets

```bash
python scripts/pull_models.py
python scripts/prefetch_strongreject.py
python scripts/prefetch_harmbench.py
python scripts/prefetch_wildguard.py
python scripts/prefetch_perplexity.py
python scripts/fetch_jailbreakbench_prompts.py
```

Some Hugging Face checkpoints require accepting their model licenses and setting `HF_TOKEN` locally.

---

## Usage

Run a single prompt:

```bash
python main.py "What is the capital of France?"
```

Run the configured prompt batch:

```bash
python main.py
```

Select an attack, defense, and judge:

```bash
python main.py \
  --attack deepinception \
  --defense self_reminder,llama_guard_output \
  --judge harmbench_mistral_7b_val_cls
```

Check pipeline wiring without model inference:

```bash
python main.py --dry-run
```

---

## Final Experiment

Run the small validation matrix first:

```powershell
.\scripts\run_final_experiment.ps1 -Quick
```

Then run the complete experiment:

```powershell
.\scripts\run_final_experiment.ps1
```

Interrupted experiments can be resumed from their checkpoint directory without rerunning completed cells.

The evaluation pipeline produces:

```text
manifest.json
evaluated_rows.csv
experiment_matrix.csv
defense_tradeoffs.csv
attack_costs.csv
defense_events.csv
llama_guard_categories.csv
audit_sample.csv
summary.md
```

These outputs separate security effectiveness, benign usability, latency, blocking behavior, and model-query cost.

---

## Tests

Run the full test suite with:

```bash
python -m unittest discover -s tests -v
```

Tests use lightweight fake models and evaluators for heavyweight paths and cover:

- component registration and routing
- attack formatting
- defense stacking
- deterministic model settings
- attack caching
- checkpointing and resume behavior
- experiment manifests
- judge wiring
- CSV evaluation
- model-memory cleanup

---

## Limitations

This study evaluates **black-box inference-time defenses** against a single canonical target model.

Results may vary across:

- model families and scales
- safety-tuning strategies
- attack budgets
- evaluator models
- random seeds

Automated safety judges can also misclassify nuanced responses. The reported results should therefore be interpreted as comparative measurements within the controlled experiment rather than universal estimates of jailbreak robustness.

---

## Academic Context & Responsible Use

This repository accompanies the paper **“Evaluating Inference-Time Defences Against Jailbreak Attacks.”**

The project evaluates existing public attacks and defenses under a controlled experimental setup. Harmful benchmark behaviors and adversarial artifacts come from published research datasets including **JailbreakBench** rather than being introduced as new harmful objectives by this project.

The system is intended for **AI security research, robustness evaluation, and defensive experimentation**.

### Contributors

- Ryan Maxin
- Vishesh Gupta
- Varun Mathur
- Adam Kaegi
