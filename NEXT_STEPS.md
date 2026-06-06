# Next Steps for the CPO/HCH Behavioral Paper

This file summarizes what changed in the repository and what should be run next.

## Current status

The repository now supports the core Paper 1 behavioral pipeline for **Criterion-Priority Obedience (CPO)** and **Hidden Criterion Hierarchies (HCH)**.

The current usable evidence is:

1. **Tier 1 controlled CPO benchmark**
   - Main causal benchmark.
   - Clean conflict/no-conflict construction.
   - Current results show a strong POR-NC vs POR-C dissociation across Qwen and Mistral.

2. **Tier 2 LLMBar-Adversarial subset**
   - Naturalistic/adversarial validation layer.
   - Current results show the same high-HHR pattern, especially clean for Qwen2.5-7B.

3. **WildBench is intentionally ignored for now**
   - The miner exists, but it is heuristic and should not be used as paper evidence until it has QC or manual verification.

## What was edited

### 1. Added strict paired-priority metrics

`src/cpo/metrics.py` now writes four additional files after `cpo metrics`:

- `paired_priority_units.csv`
- `paired_main_results.csv`
- `paired_family_breakdown.csv`
- the existing `main_results.csv` remains unchanged as a diagnostic table

The reason for this edit is important.

The old `main_results.csv` computes `POR_C` over individual order-stable priority decisions, while `HHR` and `RSR` require paired priority reversals. That is useful for debugging, but it can confuse the paper interpretation because the denominators are different.

The new `paired_main_results.csv` uses a single strict denominator:

A conflict item/paraphrase is retained only if:

1. the `a_over_b` priority condition is order-stable,
2. the `b_over_a` priority condition is order-stable,
3. both priority directions exist for the same item and paraphrase.

The headline paper table should use `paired_main_results.csv`, not the older `main_results.csv`.

### 2. Added paired metrics

The new paired metrics are:

- `PAIRED_RETENTION`: fraction of possible paired reversals that survive the strict paired filter.
- `PAIRED_POR_C`: mean priority obedience across both priority directions within retained paired units.
- `PAIRED_HHR`: whether the judge selected the same candidate after priority reversal.
- `PAIRED_RSR`: whether the judge correctly reversed priority in both directions.

## What to run next

Assuming the raw model outputs already exist under something like:

```bash
outputs/paper1_5060ti/raw/
```

run only metrics and plots first:

```bash
python -m cpo.cli metrics --config configs/paper1_5060ti.yaml
python -m cpo.cli plots --config configs/paper1_5060ti.yaml
```

Then inspect:

```bash
outputs/paper1_5060ti/tables/paired_main_results.csv
outputs/paper1_5060ti/tables/paired_family_breakdown.csv
outputs/paper1_5060ti/tables/main_results.csv
outputs/paper1_5060ti/tables/family_breakdown.csv
```

If `configs/paper1_5060ti.yaml` is not present in the repo or local machine, use the config file that produced the current `outputs/paper1_5060ti/` run. The important part is that `output_dir` must point to the existing output directory containing the raw choices.

## Recommended immediate analysis

### Step 1: Check paired headline results

Open `paired_main_results.csv` and focus first on:

- `method == direct`
- `tier == tier1`
- `tier == tier2_llmbar`
- metrics: `PAIRED_RETENTION`, `PAIRED_POR_C`, `PAIRED_HHR`, `PAIRED_RSR`

The expected pattern, if the paper claim holds, is:

- high no-conflict POR in the diagnostic table,
- low/moderate `PAIRED_POR_C`,
- high `PAIRED_HHR`,
- low `PAIRED_RSR`.

### Step 2: Keep LLMBar, ignore WildBench for now

For Paper 1, use:

- Tier 1 controlled benchmark as the main benchmark.
- LLMBar-Adversarial as the naturalistic anti-artifact validation.

Do not make WildBench a main paper claim yet. WildBench should only be used after a QC pass because the current miner uses heuristic dimension scores.

### Step 3: Run more model families

The best next experimental work is broader model-family coverage, not another benchmark yet.

Recommended priority:

1. **Llama-3.1-8B-Instruct** or **Llama-3.2-3B-Instruct**
   - Best reviewer-facing cross-family addition.
   - Requires Hugging Face access/login.

2. **Gemma-2-2B-it** or a newer Gemma instruction model if available locally
   - Useful non-Qwen, non-Mistral family.
   - May require Hugging Face license acceptance.

3. **Yi / DeepSeek / Falcon family**
   - Optional breadth after Llama/Gemma.

The goal is not to run every model. The goal is to rule out:

> This is only a Qwen/Mistral artifact.

A strong minimum model set would be:

- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct or Llama-3.2-3B-Instruct
- Gemma-2-2B-it or similar Gemma instruction model

### Step 4: Then consider another benchmark

Only after model-family coverage is stronger, add another benchmark.

The best order is:

1. Finish Tier 1 + LLMBar across more model families.
2. Add a clean human/QC-verified WildBench subset, or drop WildBench.
3. Only then consider another naturalistic benchmark.

Do not add UltraFeedback to the main paper for now. It introduces an AI-feedback criticism without adding much credibility beyond Tier 1 + LLMBar.

## Suggested paper framing after this edit

The paper should say:

> We report headline CPO/HCH results using a strict paired-priority denominator. A pair is evaluated only when both priority directions are available and both are stable under candidate-order swap. This makes POR-C, HHR, and RSR directly comparable.

Use `main_results.csv` as supplementary diagnostics and `paired_main_results.csv` as the main table.

## Checklist for the friend running next experiments

1. Pull the latest repository.
2. Install dependencies.
3. Verify GPU:

```bash
python -m cpo.cli doctor
```

4. Recompute metrics from existing raw outputs:

```bash
python -m cpo.cli metrics --config configs/paper1_5060ti.yaml
```

5. Check `paired_main_results.csv`.
6. If results look correct, run a new model family with the same benchmark and methods.
7. Do not run WildBench as main evidence until QC is done.

## Important warning

If a model has low `PAIRED_RETENTION` or low order-stable retention on LLMBar, do not overclaim from that model/dataset cell. Treat it as supportive or diagnostic only.

For the main paper, prefer cells where paired retention is reasonably high and confidence intervals are not too wide.
