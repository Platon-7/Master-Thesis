#!/usr/bin/env python3
"""Materialise Hydra-style override files for each loss variant and sbatch them.

Reads:
    configs/train_lora_base.yaml   — shared overrides
    configs/loss_sweep.yaml        — per-variant overrides

Writes:
    configs/_resolved_<id>.overrides   — one line per `key=value`
Submits:
    jobs/train_lora.job with LOSS_ID=<id>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _flatten(d: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        else:
            out.append(f"{key}={v}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path, help="configs/loss_sweep.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    base = yaml.safe_load((root / "configs" / "train_lora_base.yaml").read_text())
    sweep = yaml.safe_load(args.sweep.read_text())

    base_lines = _flatten(base)
    for variant in sweep["variants"]:
        vid = variant["id"]
        lines = base_lines + _flatten(variant["overrides"])
        out = root / "configs" / f"_resolved_{vid}.overrides"
        out.write_text("\n".join(lines) + "\n")
        print(f"[sweep] wrote {out}")

        cmd = ["sbatch", f"--export=LOSS_ID={vid}", str(root / "jobs" / "train_lora.job")]
        if args.dry_run:
            print("[sweep] would submit:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
