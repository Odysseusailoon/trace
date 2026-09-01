"""Generate an attribution report for a case from its o_t matrix + base path.

Usage: python scripts/make_report.py --case virology_5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forkscope.base_path import BasePath
from forkscope.extractor.mcq import CATEGORIES
from forkscope.report import build_report, save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eps", type=float, default=0.10)
    args = ap.parse_args()

    data = Path(args.data_dir)
    o_t = np.load(data / "aggregated" / f"o_t_{args.case}.npy")
    base = BasePath.load(data / "raw" / f"base_{args.case}.json")

    import os
    from transformers import AutoTokenizer
    snap = "/scratch/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    tok = AutoTokenizer.from_pretrained(snap if os.path.isdir(snap) else "Qwen/Qwen3-8B")

    rep = build_report(args.case, o_t, base, CATEGORIES, eps=args.eps, tokenizer=tok)
    out = save_report(rep, data / "reports")
    print(f"[report] {len(rep['forks'])} forks -> {out}/report_{args.case}.md")
    for f in rep["forks"]:
        print(f"  t={f['t']} TVD={f['tvd']:.3f}  {f['before']} -> {f['after']}")
        print(f"    ...{f['snippet'][-60:]}")


if __name__ == "__main__":
    main()
