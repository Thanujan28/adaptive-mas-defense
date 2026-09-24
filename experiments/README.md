# experiments/ — which script does what

This is the single source of truth for the scripts in `experiments/`.
There are several files here and no index elsewhere, so this document
states, for each, **what it measures**, **whether it uses the real MAS /
real models by default**, and **the exact command for a trustworthy
(non-stub) result**.

## Two rules that apply to every script here

1. **Real is the unflagged default.** A bare run uses the real
   `MASEnvironment` + `execute_task` and the real models. Stubs must be
   opted into with an explicit `--stub*` flag (matching
   `eval_detector.py` and `calibrate_detector.py`).
2. **Conditions come from `attacks/scenarios.py::IMPLEMENTED_CONDITIONS`.**
   Only `clean` and `prompt_infection` exist. `memory_poisoning` and
   `resource_exhaustion` are unimplemented stubs and are always reported
   as "NOT IMPLEMENTED - not evaluated", never assigned a number.

Model names are centralized in `security/model_names.py`
(`SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`); no
script should repeat that literal.

> **Running anything live requires Ollama + a cached MiniLM**, and a real
> episode (the `mock_calendar` tool) **mutates `configs/mock_calendar.json`**
> as a side effect. `git checkout -- configs/mock_calendar.json` before
> committing after any live run.

---

## Paper-result-producing scripts

### `eval_detector.py`
**Measures:** the security detector's AUROC, TPR@5%FPR and clean-FPR per
response and per episode, over `IMPLEMENTED_CONDITIONS` × topologies ×
seeds, against three independent label sources (`exposure`,
`outcome_judge`, and the legacy `indicator_legacy` reported only as
"circularity inflation"). Also compares the three Tier-3 investigation
strategies (formula / brute_force / no_investigation).
**Real by default:** yes — runs real episodes and the real MiniLM + real
LLM judge unless `--stub-*` is passed. Stubs are opt-in.
**Trustworthy run:**
```
python -m experiments.eval_detector
```
Writes `outputs/eval_detector_{response,episode,strategy}.csv`.

### `calibrate_detector.py`
**Measures:** false-positive rate of the observable-evidence detector on
CLEAN episodes per topology (per episode and per response), plus TPR on
the held-out payload variants, and (re)builds the per-agent clean-run
token baseline `configs/token_baseline.json`.
**Real by default:** yes — real clean episodes unless `--stub-llm` /
`--stub-encoder` are passed.
**Trustworthy run:**
```
python -m experiments.calibrate_detector
```
Writes `outputs/calibrate_detector_{episodes,responses,variants}.csv` and
`configs/token_baseline.json`.

### `ablate_signals.py`
**Measures:** AUROC / TPR@5%FPR of `security_score` per defender variant
(content-only, behaviour-only, semantic-only, fused, fused+alert,
LEAKY-baseline).
**Real by default:** for a *meaningful* ablation, use `--from-live-episodes`
(runs real episodes and extracts samples directly from them). The
`--from-jsonl-template` mode reads SYNTHETIC hand-authored samples and is
labelled as such in a loud banner and in the CSV's `# DATA SOURCE:` header.
**Trustworthy run:**
```
python -m experiments.ablate_signals --from-live-episodes --episodes 2
```
Writes `outputs/ablation_signals.csv` (prefix comment records the data
source).

### `run_experiment.py`
**Measures:** the RL/PPO side — decision steps, transition counts, PPO
security-state reproducibility, and final resource state, over
conditions × topologies × benchmark tasks. It is **not** a detector
evaluation and is **not** superseded by `eval_detector.py` (which never
touches `PPOEnvironment`). Nothing imports it; run as a script.
**Real by default:** yes — builds and steps a real `MASEnvironment` via
`PPOEnvironment`.
**Trustworthy run:**
```
python -m experiments.run_experiment
```
Writes `outputs/experiment_results.json`.

---

## Diagnostic / inspection scripts

These are viewers, not headline-number producers. They do not run the
full condition sweep; treat their output as diagnostics.

### `show_security_logs.py`
**Measures:** prints, per agent response, the observable detector
evidence, the semantic similarity fields, the fused `security_score` and
decision, the episode aggregate and the PPO state; `--tiered` adds the
per-chunk Tier 1/2/3 view.
**Real by default:** yes — a bare run executes a LIVE episode (real MAS,
real encoder) and needs Ollama. Reading pre-collected records now requires
the explicit `--from-jsonl` flag (it no longer silently prefers the demo
file). `--stub` stubs only the encoder.
**Trustworthy run:**
```
python -m experiments.show_security_logs --task "your task"
```

### `security_dashboard.py`
**Measures:** a Tkinter GUI (headless twin via `--dump`) over the tiered
observer's per-chunk Tier 1/2/3 results and the fused state.
**Real by default:** it is a VIEWER. A bare run loads a saved run
(`outputs/last_run.jsonl`) or the demo JSONL; a REAL episode requires the
explicit `--live` flag. Unlike the paper scripts this is an intentional
viewer-of-saved-data default, not a stub default — use `--live` when you
want it to run the real system.
**Trustworthy run:**
```
python -m experiments.security_dashboard --live --strategy formula
```

### `compare_subtask_reference.py`
**Measures:** old (stage label) vs new (instruction template) subtask
reference for `subtask_similarity`.
**Real by default:** **NO — synthetic only.** It does not run the MAS; it
reads a hand-authored/pre-collected JSONL (`--from-jsonl`) and prints a
loud SYNTHETIC banner. Its numbers compare references on hand-written
text, not real agent output.
**Trustworthy run:** there is no live mode; supply real agent outputs you
collected elsewhere:
```
python experiments/compare_subtask_reference.py --from-jsonl outputs/samples.jsonl
```

### `validate_pilot.py`
**Measures:** Tier-2 (NLI) and Tier-3 (judge) agreement against the
hand-labelled pilot set `tests/fixtures/judge_pilot_set.jsonl`. Rows with
`human_label: null` are SKIPPED (it never auto-generates labels).
**Real by default:** yes — loads the real NLI model and, unless
`--stub-judge`, real Ollama. Stubs are opt-in.
**Trustworthy run:**
```
python -m experiments.validate_pilot
```

### `verify_vacuity.py`
**Measures:** proves the negative-control tests in
`tests/test_no_attack_vocabulary_leak.py` are non-vacuous by temporarily
disabling each guard, confirming the control then fails, and restoring the
file byte-for-byte. It is a meta-test harness, not an experiment run.
**Real by default:** N/A (it shells out to pytest).
**Trustworthy run:**
```
python -m experiments.verify_vacuity
```

---

## Condition-iteration note

Scripts that iterate conditions MUST import
`IMPLEMENTED_CONDITIONS` from `attacks/scenarios.py` rather than a local
tuple. As of this pass, `eval_detector.py`, `run_experiment.py`,
`calibrate_semantic.py` and `ablate_signals.py` do so.