#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/smoke_test.yaml
python -m cpo.cli doctor
python -m cpo.cli build-tier1 --config "$CONFIG"
python -m cpo.cli run --config "$CONFIG" --models qwen25_3b --methods direct decomposed
python -m cpo.cli metrics --config "$CONFIG"
python -m cpo.cli plots --config "$CONFIG"
