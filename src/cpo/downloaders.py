from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils import ensure_dir, write_jsonl, atomic_write_text, truncate_text


LLMBAR_HF_ID = "princeton-nlp/LLMBar"
LLMBAR_GITHUB = "https://github.com/princeton-nlp/LLMBar.git"
WILDBENCH_HF_ID = "allenai/WildBench"
WILDBENCH_GITHUB = "https://github.com/allenai/WildBench.git"


def _safe_load_dataset(dataset_id: str, config_name: Optional[str], cache_dir: Path) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    kwargs = {"cache_dir": str(cache_dir)}
    ds = load_dataset(dataset_id, config_name, **kwargs) if config_name else load_dataset(dataset_id, **kwargs)
    rows: List[Dict[str, Any]] = []
    for split_name, split in ds.items():
        for row in split:
            d = dict(row)
            d["_hf_split"] = split_name
            d["_hf_dataset"] = dataset_id
            if config_name:
                d["_hf_config"] = config_name
            rows.append(d)
    return rows


def _git_clone(repo_url: str, dest: Path) -> None:
    if dest.exists():
        return
    ensure_dir(dest.parent)
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])


def _iter_json_files(root: Path) -> Iterable[Path]:
    for pat in ["*.json", "*.jsonl"]:
        yield from root.rglob(pat)


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["data", "examples", "instances", "rows"]:
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return [obj]
    return rows


def download_llmbar(raw_dir: Path, cache_dir: Path) -> Path:
    """Download and normalize raw LLMBar data.

    First tries Hugging Face: load_dataset('princeton-nlp/LLMBar', 'LLMBar').
    If that fails, clones the official GitHub repository and reads Dataset/**/*.json.
    """
    raw_dir = ensure_dir(raw_dir)
    cache_dir = ensure_dir(cache_dir)
    out_path = raw_dir / "llmbar_raw.jsonl"
    report_path = raw_dir / "llmbar_download_report.json"

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"source_attempts": []}

    try:
        hf_rows = _safe_load_dataset(LLMBAR_HF_ID, "LLMBar", cache_dir)
        for r in hf_rows:
            r["_source_loader"] = "huggingface"
        rows.extend(hf_rows)
        report["source_attempts"].append({"source": "huggingface", "status": "ok", "n": len(hf_rows)})
    except Exception as e:
        report["source_attempts"].append({"source": "huggingface", "status": "failed", "error": repr(e)})

    if not rows:
        repo_dir = cache_dir / "repos" / "LLMBar"
        try:
            _git_clone(LLMBAR_GITHUB, repo_dir)
            file_count = 0
            for path in _iter_json_files(repo_dir / "Dataset"):
                loaded = _load_json_or_jsonl(path)
                rel = str(path.relative_to(repo_dir))
                for r in loaded:
                    d = dict(r)
                    d["_source_loader"] = "github"
                    d["_source_path"] = rel
                    rows.append(d)
                file_count += 1
            report["source_attempts"].append({"source": "github", "status": "ok", "n": len(rows), "files": file_count})
        except Exception as e:
            report["source_attempts"].append({"source": "github", "status": "failed", "error": repr(e)})

    write_jsonl(rows, out_path)
    report["n_rows"] = len(rows)
    atomic_write_text(report_path, json.dumps(report, indent=2))
    return out_path


def download_wildbench(raw_dir: Path, cache_dir: Path) -> Path:
    """Download raw WildBench data.

    First tries Hugging Face: load_dataset('allenai/WildBench').
    If that fails, clones the official GitHub repository and reads JSON/JSONL files from eval_results/evaluation/docs when available.
    """
    raw_dir = ensure_dir(raw_dir)
    cache_dir = ensure_dir(cache_dir)
    out_path = raw_dir / "wildbench_raw.jsonl"
    report_path = raw_dir / "wildbench_download_report.json"

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"source_attempts": []}

    try:
        hf_rows = _safe_load_dataset(WILDBENCH_HF_ID, None, cache_dir)
        for r in hf_rows:
            r["_source_loader"] = "huggingface"
        rows.extend(hf_rows)
        report["source_attempts"].append({"source": "huggingface", "status": "ok", "n": len(hf_rows)})
    except Exception as e:
        report["source_attempts"].append({"source": "huggingface", "status": "failed", "error": repr(e)})

    if not rows:
        repo_dir = cache_dir / "repos" / "WildBench"
        try:
            _git_clone(WILDBENCH_GITHUB, repo_dir)
            roots = [repo_dir / "eval_results", repo_dir / "evaluation", repo_dir / "docs", repo_dir / "src"]
            file_count = 0
            for root in roots:
                if not root.exists():
                    continue
                for path in _iter_json_files(root):
                    # Ignore very small config files that cannot contain examples.
                    try:
                        loaded = _load_json_or_jsonl(path)
                    except Exception:
                        continue
                    rel = str(path.relative_to(repo_dir))
                    for r in loaded:
                        d = dict(r)
                        d["_source_loader"] = "github"
                        d["_source_path"] = rel
                        rows.append(d)
                    file_count += 1
            report["source_attempts"].append({"source": "github", "status": "ok", "n": len(rows), "files": file_count})
        except Exception as e:
            report["source_attempts"].append({"source": "github", "status": "failed", "error": repr(e)})

    write_jsonl(rows, out_path)
    report["n_rows"] = len(rows)
    atomic_write_text(report_path, json.dumps(report, indent=2))
    return out_path


def download_all(output_dir: Path, cache_dir: Path) -> Dict[str, str]:
    raw_dir = ensure_dir(output_dir / "raw_downloads")
    paths = {
        "llmbar": str(download_llmbar(raw_dir, cache_dir)),
        "wildbench": str(download_wildbench(raw_dir, cache_dir)),
    }
    return paths
