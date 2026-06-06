# CPO/HCH: Criterion-Priority Obedience in LLM-as-a-Judge

This repository contains the codebase for the Paper 1 behavioral study:

**When Rubrics Conflict: Criterion-Priority Obedience and Hidden Criterion Hierarchies in LLM Judges**

The project tests whether LLM judges obey an explicitly stated rubric priority when two evaluation criteria conflict.

The central question is:

> When a rubric says criterion A is more important than criterion B, and then the priority is reversed, does the LLM judge actually reverse its decision?

The observed failure mode is called **Hidden Criterion Hierarchy (HCH)**: a judge may continue selecting the same candidate under explicit priority reversal, suggesting that it is imposing its own latent criterion ordering instead of following the user-specified rubric.

---

## Current paper scope

For now, the paper focuses on the behavioral CPO/HCH phenomenon only.

Included:

- Tier 1 controlled synthetic CPO benchmark
- Tier 2 LLMBar-Adversarial naturalistic validation
- open-weight local judges
- direct, locked, conflict-audit, decomposed, and context-forward judging methods
- order-controlled and paired-priority metrics

Currently ignored for main claims:

- WildBench, until QC/manual verification is done
- UltraFeedback
- activation steering or mechanistic interventions

---

## Main concepts

### Criterion-Priority Obedience (CPO)

A judge is priority-obedient if it selects the candidate that wins the criterion explicitly marked as higher priority.

Example:

- Candidate 1 wins correctness.
- Candidate 2 wins fluency.
- If the rubric says correctness > fluency, Candidate 1 should win.
- If the rubric says fluency > correctness, Candidate 2 should win.

### Hidden Criterion Hierarchy (HCH)

A hidden hierarchy event occurs when the judge picks the same candidate even after the explicit priority order is reversed.

This indicates that the judge may have an internal preference such as "correctness always outranks fluency" or "format always outranks informativeness," regardless of the stated rubric.

---

## Repository structure

```text
.
├── src/cpo/
│   ├── cli.py          # command-line pipeline
│   ├── downloaders.py  # LLMBar/WildBench download helpers
│   ├── judge.py        # local open-weight A/B judge runner
│   ├── metrics.py      # OSR, POR, HHR, RSR, PSR, paired metrics
│   ├── plots.py        # figure generation
│   ├── prompts.py      # direct/locked/conflict-audit/context-forward prompts
│   ├── schema.py       # CPO item schema
│   ├── synthetic.py    # Tier 1 benchmark generator
│   ├── tier2.py        # LLMBar/WildBench CPO mining
│   └── utils.py
├── tests/
│   └── test_metrics.py
├── outputs/
│   └── paper1_5060ti/  # current result artifacts, if pushed
├── NEXT_STEPS.md       # what to run next and what changed
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -U pip
pip install -e .
```

For local GPU experiments, install a CUDA-compatible PyTorch build appropriate for your system.

The code was designed for a **16 GB GPU**, especially an RTX 5060 Ti, using 4-bit quantization through bitsandbytes.

---

## GPU check

```bash
python -m cpo.cli doctor
```

This prints Python, PyTorch, CUDA, GPU name, VRAM, transformers, and bitsandbytes information.

---

## Pipeline overview

The normal pipeline is:

```bash
python -m cpo.cli download --config <config.yaml>
python -m cpo.cli build-tier1 --config <config.yaml>
python -m cpo.cli build-tier2 --config <config.yaml>
python -m cpo.cli run --config <config.yaml>
python -m cpo.cli metrics --config <config.yaml>
python -m cpo.cli plots --config <config.yaml>
```

Or all at once:

```bash
python -m cpo.cli pipeline --config <config.yaml>
```

If raw outputs already exist and you only changed metric code, rerun only:

```bash
python -m cpo.cli metrics --config <config.yaml>
python -m cpo.cli plots --config <config.yaml>
```

---

## Important current instruction

For now, ignore WildBench as main paper evidence.

Use:

- `tier1`
- `tier2_llmbar`

Do not make claims using `tier2_wildbench` until its mined pairs are QC-checked.

---

## Outputs

After running metrics, the main tables are written to:

```text
outputs/<run_name>/tables/
```

Important files:

```text
main_results.csv                 # diagnostic metrics with mixed POR/HHR denominator
paired_main_results.csv          # preferred headline table
family_breakdown.csv             # diagnostic family-level table
paired_family_breakdown.csv      # preferred paired family-level table
order_controlled_units.csv       # order-stability filtered units
paired_priority_units.csv        # strict paired-priority units
hhr_rows.csv                     # hidden-hierarchy event rows
psr_rows.csv                     # paraphrase stability rows
raw_choices_merged.csv           # merged raw judge choices
```

### Which table should be used in the paper?

Use:

```text
paired_main_results.csv
```

for the headline CPO/HCH table.

The older:

```text
main_results.csv
```

is still useful as a diagnostic table, but it uses a different denominator for POR-C than HHR/RSR.

---

## Metrics

### OSR: Order-Stable Retention

Fraction of candidate-order-swapped units where the judge chose the same candidate identity regardless of whether that candidate appeared as Answer A or Answer B.

### POR-C: Priority Obedience Rate on conflict items

Fraction of order-stable conflict decisions where the judge selected the candidate that wins the stated higher-priority criterion.

### POR-NC: Priority Obedience Rate on no-conflict items

Same as POR-C, but for no-conflict examples where one candidate wins both criteria.

This is a competence/control check. High POR-NC with low POR-C supports the claim that failure is conflict-specific rather than general judging incompetence.

### HHR: Hidden Hierarchy Rate

Fraction of paired priority reversals where the judge selected the same candidate under both priority orders.

High HHR means the judge is not truly responding to priority reversal.

### RSR: Reversal Success Rate

Fraction of paired priority reversals where the judge correctly selected the criterion-A winner under A>B and the criterion-B winner under B>A.

### PSR: Paraphrase Stability Rate

Fraction of cases where rubric paraphrases produce the same selected candidate.

---

## Paired metric edit

The latest update adds strict paired-priority outputs.

Why this matters:

- `main_results.csv` computes POR-C over individual order-stable priority decisions.
- HHR and RSR require paired priority reversals.
- Therefore, their denominators can differ.

To avoid ambiguity, `paired_main_results.csv` keeps only cases where both priority directions exist and both are order-stable.

A paired unit survives only if:

1. `a_over_b` is order-stable,
2. `b_over_a` is order-stable,
3. both directions exist for the same item and paraphrase.

Use this strict table for paper claims.

---

## Current recommended next work

See `NEXT_STEPS.md` for detailed instructions.

Short version:

1. Rerun metrics to generate `paired_main_results.csv`.
2. Inspect direct method on `tier1` and `tier2_llmbar`.
3. Ignore WildBench for now.
4. Add another model family before adding another benchmark.

Best next model-family targets:

- Llama-3.1-8B-Instruct or Llama-3.2-3B-Instruct
- Gemma instruction model
- optionally Yi, DeepSeek, or Falcon after that

The next scientific priority is to show the phenomenon is not limited to Qwen/Mistral.

---

## Tests

Run:

```bash
pytest -q
```

The tests currently cover:

- order-stable filtering
- HHR and RSR behavior
- PSR behavior
- bootstrap CI behavior
- family-breakdown tier filtering

---

## Current caveats

1. WildBench mining is heuristic and needs QC before main-paper use.
2. LLMBar is useful naturalistic validation, but some model/dataset cells have low order-stable retention.
3. Family-level heatmaps should filter tiny denominators, e.g., do not interpret family-level HHR where `n < 30`.
4. Main paper tables should use paired metrics.

---

## License

MIT License.
