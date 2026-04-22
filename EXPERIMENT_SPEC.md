# Experiment Spec (v0): Autonomous LLM Red Teaming Research

## Goal

Turn this repo into a **controlled experiment framework** that compares vulnerability across open-source LLMs (via Ollama) under adversarial attacks that are **autonomously generated** and **consistently evaluated**.

## Core research question

Do open-source LLMs from different families show systematic differences in vulnerability to adversarial attacks, especially when prompts are autonomously generated?

## Scope (v0)

v0 is designed to be **reproducible** and **fair** by holding variables constant and starting with **Fixed attack set** mode.

- **Primary deliverables (v0)**
  - A written experiment spec (this document)
  - A repeatable run procedure
  - A consistent result schema (JSONL/JSON/SQLite-ready)
  - A comparison summary report (later step)

- **Non-goals (v0)**
  - Continuous/periodic scanning
  - Multi-turn attacks
  - Tool/RAG attacks
  - Automated mitigations

## Models under test (targets via Ollama)

Start with 3 for v0; expand after the pipeline is stable.

- **v0 targets**
  - `llama3.2:3b` (baseline)
  - `mistral:7b`
  - `gemma2:2b`

- **optional (v1+)**
  - `phi3.5:3.8b`
  - `qwen2.5:3b`

## Constants (must be held fixed for fair comparisons)

These should be treated as “locked” per experiment run. Changing any of these creates a *new* run configuration.

### Attacker (prompt generator)

From `config.yaml` (attacker section):
- **Provider**: NVIDIA NIM (OpenAI-compatible API)
- **Model**: `nvidia/llama-3.1-nemotron-ultra-253b-v1`
- **Temperature**: `0.9`
- **Max tokens**: `1024`

### Evaluator (LLM-as-judge)

From `config.yaml` (evaluator section):
- **Provider**: NVIDIA NIM (OpenAI-compatible API)
- **Model**: `nvidia/nemotron-content-safety-reasoning-4b`
- **Temperature**: `0.1`
- **Max tokens**: `512`

### Target generation parameters (Ollama)

From `config.yaml` (target section):
- **Temperature**: `0.7`
- **Max tokens**: `512` (Ollama `num_predict`)
- **Timeout**: `30s`

### Orchestrator/run parameters

From `config.yaml` (orchestrator section):
- **max_attacks_per_strategy**: `5`
- **delay_between_attacks**: `0.5s`
- **output_dir**: `reports/`
- **retry_attempts**: present in config but **not wired in code yet** (treat as “unused” in v0)

## Attack strategies (kept constant across targets)

From `config.yaml` (strategies section):

- **jailbreak**
  - techniques: `role_play`, `hypothetical`, `many_shot`
  - variants_per_run: `5`

- **prompt_injection**
  - techniques: `direct_override`, `delimiter_confusion`
  - variants_per_run: `4`

- **data_extraction**
  - techniques: `system_prompt_leak`, `model_metadata`
  - variants_per_run: `4`

- **bias_probe**
  - axes: `gender`, `race`, `religion`, `political`
  - variants_per_run: `4`

## Two evaluation modes

### Mode A: Fixed attack set (v0)

**Purpose:** Fair comparison across models.

- Generate prompts **once** for each `(attack_type, technique_or_axis)` at the start.
- Save them as a **dataset artifact** with stable `prompt_id`s.
- Replay the same dataset across every target model.

**Notes about current code:** `main.py` currently generates prompts inline per strategy; the v0 pipeline will split this into “dataset generation” then “replay”.

### Mode B: Adaptive attacks (v1+)

**Purpose:** Simulate realistic adversarial behavior.

- Generate prompts **per target model**, optionally conditioned on:
  - target identity (optional)
  - prior refusals/responses (recommended if you do multi-round)

This mode is inherently less “paired” and requires extra care in analysis.

## Output schema (per attempt)

Each result row MUST include:

- `target_model` (e.g. `mistral:7b`)
- `attack_type` (e.g. `prompt_injection`)
- `technique` (or axis for bias probes)
- `prompt_id` (stable identifier; crucial for fixed-mode paired comparisons)
- `prompt`
- `response`
- `success` (bool)
- `severity` (`none|low|medium|high|critical`)
- `policy_breach_type` (e.g. `harmful_compliance`, `system_prompt_leak`, `data_leakage`, `bias`, `none`, etc.)
- `evidence_span` (short excerpt/quote or justification)
- `reasoning` (short text)

Strongly recommended additional fields (for reproducibility & debugging):

- `run_id` (uuid)
- `mode` (`fixed` or `adaptive`)
- `dataset_id` (for fixed mode)
- `attacker_model`, `attacker_temperature`, `attacker_max_tokens`
- `evaluator_model`, `evaluator_temperature`
- `target_temperature`, `target_max_tokens`, `target_timeout`
- `raw_score`, `confidence`
- `timestamp` (attempt time)
- `error` (if request failed; do not silently drop failures)
- `raw_judge_output` (store full judge JSON when available)

## Metrics to track (v0)

Minimum quantitative metrics:
- **Overall success rate** per target model
- Success rate by:
  - `attack_type`
  - `technique`/axis
- **Severity distribution** per model (counts by severity)
- **Policy breach type distribution** per model

Qualitative outputs:
- Top successful examples per model (ranked by severity then confidence), including:
  - prompt, response excerpt, evidence_span, reasoning

Fixed-mode paired metrics (high value):
- For each `prompt_id`, which models succeeded vs failed
- “Win/loss” style comparisons between model pairs

## Reproducibility requirements

For each experimental run, record:
- The exact `config.yaml` values used (embed in results, or store a config snapshot)
- Target model list and order
- Dataset artifact id and creation timestamp (fixed mode)
- Ollama endpoint (`base_url`) used
- Attacker/evaluator model ids and parameters

## Known v0 caveats (documented, not solved here)

- `main.py` contains hardcoded scenario text used for testing:
  - jailbreak topic string
  - prompt-injection system prompt
  - data-extraction “secret” system prompt
  These should be moved into a scenario/dataset config before you claim broad generality.

- `orchestrator.retry_attempts` is not used by code yet; runs may fail transiently.

- Target-side stochasticity (temperature) means exact outputs may not be identical run-to-run; focus on aggregate metrics + paired comparisons.

## Run procedure (v0 checklist)

1) Pull target models in Ollama:
   - `ollama pull llama3.2:3b`
   - `ollama pull mistral:7b`
   - `ollama pull gemma2:2b`

2) Ensure secrets exist:
   - `.env` contains `NVIDIA_API_KEY=...` (do not commit)

3) Fixed-mode dataset generation:
   - Generate prompts once per `(attack_type, technique_or_axis)` and save dataset artifact.

4) Replay dataset across targets:
   - For each target model, run all prompts, evaluate, and store results.

5) Produce summary:
   - Compute metrics + qualitative examples.
   - Save analysis output (Markdown + JSON).

