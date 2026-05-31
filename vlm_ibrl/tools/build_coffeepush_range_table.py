"""
Rebuild the CoffeePush range/separation table across Robometer-FT checkpoints.

Reads cm_robometer_*_CoffeePush.json from /scratch-shared/$USER/vlm_ibrl_cm/,
filters by model_path substring, and prints a multi-column markdown table:
4B baseline, FT step-3000, step-4000, step-5000 (whichever exist).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CM_BASE = Path("/scratch-shared/pkarageorgis1/vlm_ibrl_cm")


def find_latest_cm(model_substr: str, task: str = "CoffeePush") -> Path | None:
    matches: list[tuple[float, Path]] = []
    if not CM_BASE.exists():
        return None
    for jobdir in CM_BASE.iterdir():
        if not jobdir.is_dir():
            continue
        for p in jobdir.glob(f"cm_robometer_*_{task}.json"):
            try:
                d = json.loads(p.read_text())
                if model_substr in d.get("args", {}).get("model_path", ""):
                    matches.append((p.stat().st_mtime, p))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def stats(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = d["results"]
    sp = np.array([r["success_prob"] for r in rows])
    pr = np.array([r["progress_reward"] for r in rows])
    gt = np.array([r["gt"] for r in rows])
    pos = sp[gt == 1]
    neg = sp[gt == 0]
    return {
        "n": len(rows),
        "sp_range": float(sp.max() - sp.min()),
        "sp_min": float(sp.min()),
        "sp_max": float(sp.max()),
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "separation": float(pos.mean() - neg.mean()),
        "pr_min": float(pr.min()),
        "pr_max": float(pr.max()),
        "tpr": d.get("confusion_matrix", {}).get("tpr"),
        "fpr": d.get("confusion_matrix", {}).get("fpr"),
    }


def main() -> None:
    columns = [
        ("Robometer-4B", find_latest_cm("Robometer-4B")),
        ("Robometer-FT step-3000", find_latest_cm("Robometer_FT_consolidated/run1_icl_ours_step3000")),
        ("Robometer-FT step-4000", find_latest_cm("Robometer_FT_consolidated/run1_icl_ours_step4000")),
        ("Robometer-FT step-5000", find_latest_cm("Robometer_FT_consolidated/run1_icl_ours_step5000")),
        ("Qwen3.5-FT step-3000",   find_latest_cm("Qwen35_FT_consolidated/run4_step3000")),
        ("Qwen3.5-FT step-4000",   find_latest_cm("Qwen35_FT_consolidated/run4_step4000")),
        ("Qwen3.5-FT step-5000",   find_latest_cm("Qwen35_FT_consolidated/run4_step5000")),
    ]

    resolved = []
    for label, path in columns:
        if path is None or not path.exists():
            print(f"[skip] {label}: dump not found yet", file=sys.stderr)
            continue
        resolved.append((label, stats(path), path))

    if len(resolved) < 2:
        print("Need at least 2 columns to render table", file=sys.stderr)
        sys.exit(1)

    rows = [
        ("success_prob overall range", lambda s: f"{s['sp_range']:.4f} wide"),
        ("success_prob min / max",     lambda s: f"{s['sp_min']:.4f} / {s['sp_max']:.4f}"),
        ("GT=1 success clips — mean",  lambda s: f"{s['pos_mean']:.3f}"),
        ("GT=0 failure clips — mean",  lambda s: f"{s['neg_mean']:.3f}"),
        ("pos − neg separation",       lambda s: f"{s['separation']:.3f}"),
        ("progress range",             lambda s: f"[{s['pr_min']:.2f}, {s['pr_max']:.2f}]"),
        ("TPR @ τ=0.5",                lambda s: f"{s['tpr']:.3f}" if s['tpr'] is not None else "—"),
        ("FPR @ τ=0.5",                lambda s: f"{s['fpr']:.3f}" if s['fpr'] is not None else "—"),
    ]

    headers = ["Metric"] + [c[0] for c in resolved]
    print("|", " | ".join(headers), "|")
    print("|", " | ".join(["---"] * len(headers)), "|")
    for label, getter in rows:
        cells = [label] + [getter(c[1]) for c in resolved]
        print("|", " | ".join(cells), "|")

    print()
    print("Sources:")
    for label, _, path in resolved:
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
