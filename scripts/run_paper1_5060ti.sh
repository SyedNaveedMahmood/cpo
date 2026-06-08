#!/usr/bin/env bash
# Full paper1 run on RTX 5060 Ti 16 GB.
# Models are run sequentially (one at a time) because each 7/8B model at
# 4-bit occupies ~5-6 GiB — two models simultaneously would OOM.
set -euo pipefail

CONFIG=${1:-configs/paper1_5060ti.yaml}

echo "=== CPO/HCH Paper 1 pipeline on RTX 5060 Ti ==="
echo "Config: $CONFIG"

# Environment check
python -m cpo.cli doctor

# Download external benchmarks (LLMBar, WildBench from HuggingFace).
# Pass --skip-download to this script to skip if already cached.
python -m cpo.cli download --config "$CONFIG"

# Build Tier 1 benchmark (fully local, no GPU needed).
python -m cpo.cli build-tier1 --config "$CONFIG"

# Build Tier 2 CPO-mined subsets.
# Check outputs/paper1_5060ti/data/*.diagnostics.json for yield.
# If llmbar writes 0 items, set llmbar_require_adversarial_filter: false
# in the config and rerun this step.
python -m cpo.cli build-tier2 --config "$CONFIG"

# NOTE: Llama-3.1-8B-Instruct is a gated model.
# Before running, authenticate with:  huggingface-cli login
# or set:  export HF_TOKEN=<your_token>
# The CLI will error clearly if the token is missing.

# Run all methods for Qwen 2.5-3B
python -m cpo.cli run --config "$CONFIG" \
    --models qwen25_3b \
    --methods direct locked conflict_audit decomposed context_forward

# Run all methods for Qwen 2.5-7B
python -m cpo.cli run --config "$CONFIG" \
    --models qwen25_7b \
    --methods direct locked conflict_audit decomposed context_forward

# Run all methods for Mistral-7B-v0.3
python -m cpo.cli run --config "$CONFIG" \
    --models mistral7b_v03 \
    --methods direct locked conflict_audit decomposed context_forward

# Run all methods for Llama-3.1-8B (requires HF token)
python -m cpo.cli run --config "$CONFIG" \
    --models llama31_8b \
    --methods direct locked conflict_audit decomposed context_forward

# Compute all metrics (OSR, POR-C, POR-NC, HHR, RSR, PSR with 95% CIs).
python -m cpo.cli metrics --config "$CONFIG"

# Generate all paper figures.
python -m cpo.cli plots --config "$CONFIG"

echo ""
echo "=== Done. Results in: $(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output_dir'])")/tables/ ==="
