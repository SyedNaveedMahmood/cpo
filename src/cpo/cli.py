from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .downloaders import download_all
from .metrics import compute_all_metrics
from .plots import make_all_plots
from .synthetic import generate_tier1
from .tier2 import merge_benchmarks, normalize_llmbar, normalize_wildbench
from .utils import disable_proxy_env, ensure_dir, gpu_memory_used_mb, load_config, read_jsonl, set_seed, write_jsonl


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    out = Path(cfg["output_dir"])
    return {
        "out": out,
        "data": out / "data",
        "raw": out / "raw",
        "tables": out / "tables",
        "figures": out / "figures",
        "cache": Path(cfg.get("cache_dir", "hf_cache")),
    }


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("GPU memory used MB:", gpu_memory_used_mb())
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            print(f"  VRAM total: {props.total_memory / 1024**3:.1f} GiB")
            print(f"  CUDA capability: sm_{props.major}{props.minor}")
    except Exception as e:
        print("torch check failed:", repr(e))
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers check failed:", repr(e))
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", getattr(bnb, "__version__", "unknown"))
    except Exception as e:
        print("bitsandbytes check failed:", repr(e))


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    disable_proxy_env(bool(cfg.get("allow_proxy", False)))
    paths = resolve_paths(cfg)
    ensure_dir(paths["data"])
    downloaded = download_all(paths["data"], paths["cache"])
    print(json.dumps(downloaded, indent=2))


# ---------------------------------------------------------------------------
# build-tier1
# ---------------------------------------------------------------------------

def cmd_build_tier1(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    disable_proxy_env(bool(cfg.get("allow_proxy", False)))
    set_seed(int(cfg["seed"]))
    paths = resolve_paths(cfg)
    ensure_dir(paths["data"])
    data_cfg = cfg["data"]
    families = list(data_cfg["families"])
    # Safety family is INCLUDED by default (it is in the proposal's family list).
    # If you need to exclude it (e.g., for a fast ablation), set
    # include_safety_family: false in the config.
    if not data_cfg.get("include_safety_family", True):
        families = [f for f in families if f != "safety_vs_helpfulness"]
    conflict, no_conflict = generate_tier1(
        seed=int(cfg["seed"]),
        families=families,
        conflict_per_family=int(data_cfg["tier1_conflict_per_family"]),
        no_conflict_each_side_per_family=int(data_cfg["tier1_no_conflict_per_family_each_side"]),
    )
    c_path = paths["data"] / "tier1_conflict.jsonl"
    nc_path = paths["data"] / "tier1_no_conflict.jsonl"
    write_jsonl([x.to_dict() for x in conflict], c_path)
    write_jsonl([x.to_dict() for x in no_conflict], nc_path)
    merge_benchmarks([c_path, nc_path], paths["data"] / "tier1_all.jsonl")
    print(f"wrote {len(conflict)} conflict items -> {c_path}")
    print(f"wrote {len(no_conflict)} no-conflict items -> {nc_path}")


# ---------------------------------------------------------------------------
# build-tier2
# ---------------------------------------------------------------------------

def cmd_build_tier2(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    disable_proxy_env(bool(cfg.get("allow_proxy", False)))
    paths = resolve_paths(cfg)
    ensure_dir(paths["data"])
    raw_llmbar = paths["data"] / "raw_downloads" / "llmbar_raw.jsonl"
    raw_wildbench = paths["data"] / "raw_downloads" / "wildbench_raw.jsonl"
    if not raw_llmbar.exists() or not raw_wildbench.exists():
        print("Raw benchmark files missing; running downloader first.")
        download_all(paths["data"], paths["cache"])
    out_llmbar = paths["data"] / "tier2_llmbar_cpo.jsonl"
    out_wildbench = paths["data"] / "tier2_wildbench_cpo.jsonl"

    # Pass require_adversarial flag from config.  This is False when the HF
    # download flattens all splits into a single stream (no split metadata),
    # which is common with some dataset versions.  Check the diagnostics JSON
    # to verify actual filter yield.
    require_adversarial = bool(cfg.get("llmbar_require_adversarial_filter", True))
    n1 = normalize_llmbar(
        raw_llmbar,
        out_llmbar,
        int(cfg["data"]["tier2_llmbar_target_n"]),
        int(cfg["seed"]),
        require_adversarial=require_adversarial,
    )
    n2 = normalize_wildbench(
        raw_wildbench,
        out_wildbench,
        int(cfg["data"]["tier2_wildbench_target_n"]),
        int(cfg["seed"]),
    )

    print(f"wrote {n1} LLMBar CPO-mined items -> {out_llmbar}")
    if n1 == 0:
        print(
            "  WARNING: 0 LLMBar items written.  If you used load_dataset() and "
            "the split metadata was flattened, set llmbar_require_adversarial_filter: false "
            "in your config and rerun build-tier2."
        )

    print(f"wrote {n2} WildBench CPO-mined items -> {out_wildbench}")
    if n2 == 0:
        print(
            "  WARNING: 0 WildBench items written.  The miner requires at least 2 "
            "candidate responses per row.  Check wildbench_raw.jsonl schema."
        )

    available = [
        p
        for p in [paths["data"] / "tier1_all.jsonl", out_llmbar, out_wildbench]
        if p.exists() and p.stat().st_size > 50
    ]
    if available:
        n = merge_benchmarks(available, paths["data"] / "merged_all.jsonl")
        print(f"wrote merged benchmark with {n} rows -> {paths['data'] / 'merged_all.jsonl'}")


# ---------------------------------------------------------------------------
# Model config builder
# ---------------------------------------------------------------------------

def _make_model_config(cfg: Dict[str, Any], model_tag: str):
    m = cfg["models"][model_tag]
    inf = cfg["inference"]
    cache_dir = str(Path(cfg.get("cache_dir", "hf_cache")) / "models" / model_tag)
    from .judge import ModelConfig
    return ModelConfig(
        tag=model_tag,
        name=m["name"],
        trust_remote_code=bool(m.get("trust_remote_code", True)),
        load_in_4bit=bool(inf.get("load_in_4bit", True)),
        torch_dtype=str(inf.get("torch_dtype", "float16")),
        gpu_max_memory=str(inf.get("gpu_max_memory", "15GiB")),
        cpu_max_memory=str(inf.get("cpu_max_memory", "48GiB")),
        max_length=int(inf.get("max_length", 1536)),
        batch_size=int(inf.get("batch_size", 4)),
        cache_dir=cache_dir,
    )


def _check_hf_login_if_required(m: Dict[str, Any], model_tag: str) -> None:
    """Check that HF_TOKEN is set for gated models (e.g., Llama 3.x)."""
    if not m.get("requires_hf_login", False):
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError(
            f"Model '{model_tag}' ({m['name']}) requires a Hugging Face token. "
            "Please run:\n"
            "  huggingface-cli login\n"
            "or set the HF_TOKEN environment variable before running experiments."
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from .judge import (
        ABJudge,
        load_items,
        run_context_forward,
        run_decomposed,
        run_direct_like,
        run_preflight,
        save_dataframe_jsonl,
    )
    cfg = load_config(args.config)
    disable_proxy_env(bool(cfg.get("allow_proxy", False)))
    set_seed(int(cfg["seed"]))
    paths = resolve_paths(cfg)
    ensure_dir(paths["raw"])

    # Resolve benchmark data path.
    data_path = Path(args.data) if args.data else paths["data"] / "merged_all.jsonl"
    if not data_path.exists():
        fallbacks = [
            paths["data"] / "tier1_all.jsonl",
            paths["data"] / "tier1_conflict.jsonl",
        ]
        data_path = next((p for p in fallbacks if p.exists()), data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Benchmark data not found. Run build-tier1 / build-tier2 first. "
            f"Missing: {data_path}"
        )
    items = load_items([data_path])
    print(f"Loaded {len(items)} CPO items from {data_path}")

    model_tags = args.models or list(cfg["models"].keys())
    methods = args.methods or list(cfg.get("methods", ["direct"]))

    for model_tag in model_tags:
        if model_tag not in cfg["models"]:
            raise KeyError(f"Model tag not in config: {model_tag}")
        m_cfg_raw = cfg["models"][model_tag]
        _check_hf_login_if_required(m_cfg_raw, model_tag)
        mcfg = _make_model_config(cfg, model_tag)
        judge = ABJudge(mcfg)
        model_out = paths["raw"] / model_tag
        ensure_dir(model_out)
        try:
            print(f"\n{'='*60}")
            print(f"Loading {model_tag}: {mcfg.name}")
            judge.load()
            (model_out / "scorer_info.json").write_text(
                json.dumps(judge.scorer_info(), indent=2), encoding="utf-8"
            )

            pre = run_preflight(
                judge,
                int(cfg["inference"]["n_preflight"]),
                mcfg.batch_size,
            )
            pre.to_csv(model_out / "preflight_rows.csv", index=False)
            acc = float(pre["is_correct"].mean())
            print(f"{model_tag} preflight accuracy: {acc:.3f}")
            if acc < float(cfg["inference"]["min_preflight_acc"]):
                (model_out / "preflight_failed.json").write_text(
                    json.dumps({"preflight_acc": acc}, indent=2), encoding="utf-8"
                )
                print(
                    f"SKIPPING {model_tag}: preflight accuracy {acc:.3f} below "
                    f"threshold {cfg['inference']['min_preflight_acc']}. "
                    f"This model will be excluded from paper analysis."
                )
                continue

            for method in methods:
                method_out = model_out / method
                ensure_dir(method_out)
                raw_path = method_out / "raw_choices.jsonl"
                if raw_path.exists() and not args.force:
                    print(f"  Skipping existing: {raw_path}")
                    continue
                print(f"  Running {model_tag}/{method}")

                if method in {"direct", "locked", "conflict_audit"}:
                    df = run_direct_like(
                        judge,
                        items,
                        method,
                        int(cfg["inference"].get("paraphrases", 2)),
                        mcfg.batch_size,
                    )
                elif method == "decomposed":
                    df = run_decomposed(judge, items, mcfg.batch_size)
                    # Save criterion-level rows separately (needed for paper's
                    # decomposed analysis and reproducibility).
                    crit = df.attrs.get("criterion_rows")
                    if isinstance(crit, pd.DataFrame) and len(crit) > 0:
                        crit.to_csv(method_out / "criterion_rows.csv", index=False)
                        save_dataframe_jsonl(crit, method_out / "criterion_rows.jsonl")
                elif method == "context_forward":
                    # context_forward runs its own internal decomposed pass.
                    # We do NOT run decomposed separately here to avoid double
                    # inference.  If you want decomposed results saved, run
                    # the decomposed method as a separate method entry.
                    df = run_context_forward(judge, items, mcfg.batch_size)
                else:
                    raise ValueError(f"Unknown method: {method}")

                df["model_tag"] = model_tag
                df["model_name"] = mcfg.name
                save_dataframe_jsonl(df, raw_path)
                df.to_csv(method_out / "raw_choices.csv", index=False)
                print(f"  wrote {len(df)} rows -> {raw_path}")

        finally:
            judge.unload()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def cmd_metrics(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    raw_paths = list(paths["raw"].glob("*/*/raw_choices.jsonl"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw choices found under {paths['raw']}")
    tables = paths["tables"]
    out = compute_all_metrics(
        raw_paths,
        tables,
        n_boot=int(cfg["metrics"].get("bootstrap_samples", 10000)),
        alpha=float(cfg["metrics"].get("ci_alpha", 0.05)),
        seed=int(cfg["seed"]),
    )
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def cmd_plots(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    made = make_all_plots(paths["tables"], paths["figures"])
    print(json.dumps({k: str(v) for k, v in made.items()}, indent=2))


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    if not args.skip_download:
        cmd_download(args)
    cmd_build_tier1(args)
    cmd_build_tier2(args)
    cmd_run(args)
    cmd_metrics(args)
    cmd_plots(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cpo", description="CPO/HCH Paper 1 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    for name, func in [
        ("download", cmd_download),
        ("build-tier1", cmd_build_tier1),
        ("build-tier2", cmd_build_tier2),
        ("metrics", cmd_metrics),
        ("plots", cmd_plots),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.set_defaults(func=func)

    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--data", default=None)
    run_p.add_argument("--models", nargs="*", default=None)
    run_p.add_argument("--methods", nargs="*", default=None)
    run_p.add_argument("--force", action="store_true",
                       help="Re-run and overwrite existing raw_choices.jsonl files.")
    run_p.set_defaults(func=cmd_run)

    pipe_p = sub.add_parser("pipeline")
    pipe_p.add_argument("--config", required=True)
    pipe_p.add_argument("--data", default=None)
    pipe_p.add_argument("--models", nargs="*", default=None)
    pipe_p.add_argument("--methods", nargs="*", default=None)
    pipe_p.add_argument("--force", action="store_true")
    pipe_p.add_argument("--skip-download", action="store_true")
    pipe_p.set_defaults(func=cmd_pipeline)
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
