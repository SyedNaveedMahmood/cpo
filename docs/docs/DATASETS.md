# Dataset sources

## Tier 1
Generated locally by `cpo.synthetic`. It produces controlled criterion-conflict pairs and no-conflict controls.

## Tier 2A: LLMBar-Adversarial
The downloader tries:

```python
load_dataset("princeton-nlp/LLMBar", "LLMBar")
```

If unavailable, it clones the official GitHub repository and reads `Dataset/LLMBar/Adversarial/*/dataset.json` plus other JSON files.

The normalizer uses LLMBar's gold label as the instruction-following/correctness winner and the opposite output as the adversarial surface-quality winner.

## Tier 2B: WildBench
The downloader tries:

```python
load_dataset("allenai/WildBench")
```

If unavailable, it clones the official GitHub repository and scans JSON/JSONL artifacts. WildBench formats may vary; the miner is conservative and writes diagnostics if it cannot expose response pairs and dimension/checklist evidence.
