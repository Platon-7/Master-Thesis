"""Extract specific scalar values at requested step grid for evidence-based analysis."""
import os, csv
from collections import defaultdict
import numpy as np
import wandb

OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
RUNS = {"loss1": "nlp-squad/Robometer_LoRA/rl5u04gb",
        "loss2": "nlp-squad/Robometer_LoRA/p4z06ajv"}
TARGET_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500]
EVAL_KEY = "eval_p_rank/kendall_avg_robometer_frames_eval_failsafe"


def get_series(rows, key):
    """Return sorted [(step, value)] for non-null values of key."""
    out = []
    for r in rows:
        v = r.get(key); s = r.get("_step")
        if v is not None and s is not None and not isinstance(v, dict):
            out.append((int(s), float(v)))
    return sorted(out)


def value_at(series, target_step, smooth_radius=20):
    """Return mean value within [target-smooth_radius, target+smooth_radius] window.
    Falls back to nearest single value if no points in window."""
    if not series:
        return None
    vals = [v for s, v in series if abs(s - target_step) <= smooth_radius]
    if vals:
        return float(np.mean(vals))
    # Fallback to nearest
    nearest = min(series, key=lambda sv: abs(sv[0] - target_step))
    return nearest[1]


def main():
    print("Fetching wandb runs ...")
    api = wandb.Api(timeout=60)
    runs = {m: list(api.run(p).scan_history(page_size=2000)) for m, p in RUNS.items()}

    metrics = {
        "train_prog_loss": "train/prog_loss",
        "train_success_loss": "train/success_loss",
        "preclip_grad_norm": "optim/preclip_grad_norm",
        "train_spearman_corr": "train/spearman_corr",
    }

    print("\n" + "=" * 80)
    print("TRAINING TRAJECTORY (smoothed ±20 step window)")
    print("=" * 80)
    rows_csv = []
    for m, hist in runs.items():
        print(f"\n--- {m.upper()} ---")
        print(f"{'step':>6}  " + "  ".join(f"{k:>20}" for k in metrics.keys()))
        for step in TARGET_STEPS:
            vals = []
            for label, key in metrics.items():
                ser = get_series(hist, key)
                v = value_at(ser, step)
                vals.append(v if v is not None else float("nan"))
            print(f"{step:>6}  " + "  ".join(f"{v:>20.5f}" for v in vals))
            rows_csv.append([m, step] + vals)

    # Validation Kendall trajectory at every eval round (Failsafe)
    print("\n" + "=" * 80)
    print("FAILSAFE EVAL KENDALL (every eval round, raw not smoothed)")
    print("=" * 80)
    eval_rows = []
    for m, hist in runs.items():
        print(f"\n--- {m.upper()} ---")
        ser = get_series(hist, EVAL_KEY)
        # Also pull droid, robometer, metaworld for completeness
        per_split = {}
        for sp in ["eval_droid", "eval_robometer", "eval_metaworld", "eval_failsafe"]:
            per_split[sp] = get_series(hist, f"eval_p_rank/kendall_avg_robometer_frames_{sp}")
        # Pull steps from failsafe (canonical eval cadence)
        canonical_steps = [s for s, v in per_split["eval_failsafe"]]
        print(f"{'step':>6}  {'droid':>9}  {'robometer':>10}  {'metaworld':>10}  {'failsafe':>10}")
        for step in canonical_steps:
            row_vals = []
            for sp in ["eval_droid", "eval_robometer", "eval_metaworld", "eval_failsafe"]:
                series = per_split[sp]
                # Find value at this exact step
                v = next((vv for ss, vv in series if ss == step), None)
                row_vals.append(v)
            print(f"{step:>6}  " + "  ".join(f"{v:>9.4f}" if v is not None else "      —   " for v in row_vals))
            eval_rows.append([m, step] + row_vals)

    # Save CSVs
    train_csv = f"{OUT}/table_11_training_grid.csv"
    eval_csv = f"{OUT}/table_12_eval_kendall_per_split.csv"
    with open(train_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "step"] + list(metrics.keys()))
        w.writerows(rows_csv)
    with open(eval_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "step"] + ["kendall_droid", "kendall_robometer",
                                         "kendall_metaworld", "kendall_failsafe"])
        w.writerows(eval_rows)
    print(f"\nWrote {train_csv}")
    print(f"Wrote {eval_csv}")


if __name__ == "__main__":
    main()
