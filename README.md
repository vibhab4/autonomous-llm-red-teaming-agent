# Autonomous LLM Red Teaming Research

A controlled **experiment framework** for comparing vulnerability across open-source LLMs (via **Ollama**) under adversarial attacks that are **autonomously generated** (NVIDIA NIM “attacker”) and **consistently evaluated** (NVIDIA NIM “judge”).

The source-of-truth experiment design lives in [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md).

## Architecture

```
NIM (attacker) → generates adversarial prompts
        ↓
Ollama (target) → responds
        ↓
NIM (evaluator) → judges if attack succeeded
        ↓
Structured results → per-model + cross-model comparison
```

Default model IDs are defined in `config.yaml` (see [Tech stack](#tech-stack)).

## Experiment workflow (v0)

v0 is designed for **fair model-to-model comparison** using a **Fixed attack set**:

1) **Generate a prompt dataset** once (prompts are reused across all targets)
2) **Replay the dataset** across each target model via Ollama
3) **Evaluate and aggregate** into per-model summaries and cross-model comparisons

Adaptive attacks (prompts generated per-model, optionally multi-round) are a v1+ goal.

### Fixed-mode commands (current)

1) Generate a fixed prompt dataset:

```bash
python generate_dataset.py --n-per-combo 1
```

2) Replay that dataset across target models (Ollama) and evaluate each attempt:

```bash
python run_fixed_experiment.py \
  --dataset datasets/attacks_<dataset_id>.jsonl \
  --models llama3.2:3b,mistral:7b,gemma2:2b
```

This writes per-attempt JSONL to `results/fixed_results_<run_id>.jsonl` by default.

## Setup

### 1. Clone and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Ollama (local target model)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2:3b
ollama serve          # Start the local server
```

### 3. Get your NVIDIA NIM API key

- Go to https://build.nvidia.com
- Sign up for a free account
- Generate an API key

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY to your key
```

If a key was ever committed to a repo or shared, **revoke it** in the NVIDIA dashboard and create a new one.

### 5. Run the agent

```bash
python main.py
```

Note: `main.py` currently runs a single configured target and generates prompts inline. The repo is being evolved into the multi-model experiment workflow described above; see [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md).

## Project structure

Python modules live at the **repository root** (there is no `agent/` subfolder):

```
red-team-agent/
├── attacker.py       # NIM-powered attack prompt generator
├── target.py         # Ollama local model wrapper
├── evaluator.py      # LLM-as-a-judge result scorer
├── main.py           # Orchestrator — entry point
├── config.yaml       # Model URLs, strategies, orchestrator settings
├── reports/          # Generated reports (created automatically)
├── requirements.txt
└── .env.example
```

## Attack strategies

| Strategy | Goal | Techniques / axes |
|----------|------|---------------------|
| `jailbreak` | Bypass safety training | `role_play`, `hypothetical`, `many_shot` |
| `prompt_injection` | Override system prompt | `direct_override`, `delimiter_confusion` |
| `data_extraction` | Leak hidden info | `system_prompt_leak`, `model_metadata` |
| `bias_probe` | Surface unfair or inconsistent outputs | Axes: `gender`, `race`, `religion`, `political` |

## Configuration

Edit `config.yaml` to:

- Swap the **target** model (`ollama pull …` then update `target.model`) for single-model runs
- Disable strategies (`enabled: false`)
- Adjust `variants_per_run` and orchestrator limits
- Point **attacker** / **evaluator** at different NIM models or endpoints

### Orchestrator

- **`max_attacks_per_strategy`** — caps how many prompt variants run per strategy (combined with `variants_per_run`).
- **`delay_between_attacks`** — seconds between attempts (rate limiting).
- **`output_dir`** — where JSON and Markdown reports are written.
- **`retry_attempts`** — present in config for future use; **not read by the app** today. API clients may still apply their own retry behavior.

## Outputs and operator notes

- **Terminal:** Timestamped logs, per-attempt summaries, and (if `rich` is installed) a colored results table. Without `rich`, a plain-text summary is printed.
- **Reports:** Current implementation writes `report_<timestamp>.json` and `report_<timestamp>.md` under the configured output directory (default `reports/`).
- **Experiments (planned):** dataset artifact(s) + per-model results + comparison summary (see `EXPERIMENT_SPEC.md`).
- **Evaluator:** The judge may print `[DEBUG]` lines while parsing verdicts; this is normal for troubleshooting.

## Tech stack

- **Attacker:** `nvidia/llama-3.1-nemotron-ultra-253b-v1` via NVIDIA NIM (OpenAI-compatible API) — configurable in `config.yaml`
- **Evaluator:** `nvidia/nemotron-content-safety-reasoning-4b` via NVIDIA NIM — configurable in `config.yaml`
- **Target:** e.g. **Llama 3.2 3B** (`llama3.2:3b`) via Ollama — configurable in `config.yaml`
- **Python** — orchestration
- **LLM-as-a-judge** — automated evaluation with local guardrails in `evaluator.py`
