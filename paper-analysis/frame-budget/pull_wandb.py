"""Pull the two Figure 5.3 metrics from a Robometer_FT W&B run.

Panel (a)  eval_p_rank/max_succ_fail_diff_last/<split>   the widest success-failure
           margin over that split's tasks, exactly the panel's quantity.
Panel (b)  eval_p_rank/ranking_acc_last/robometer_frames_eval_droid

run1 trained with ICL so its keys carry _iclon/_icloff; run2 does not, so its keys
are unsuffixed. SUFFIX picks between them ("" for run2, "_iclon" for run1).

  python pull_wandb.py <wandb_id> <out_label> [suffix]
"""
import json, sys, wandb

SPLITS = ["droid", "robometer", "metaworld", "failsafe"]

def keys(sfx):
    ks = [f"eval_p_rank/max_succ_fail_diff_last/robometer_frames_eval_{s}{sfx}" for s in SPLITS]
    ks += [f"eval_p_rank/ranking_acc_last/robometer_frames_eval_{s}{sfx}" for s in SPLITS]
    return ks

def main(rid, label, sfx=""):
    api = wandb.Api(timeout=90)
    r = api.run(f"nlp-squad/Robometer_FT/{rid}")
    K = keys(sfx)
    rows = []
    for x in r.scan_history(keys=["_step"] + K, page_size=2000):
        row = {k: v for k, v in x.items() if v is not None}
        if len(row) > 1:
            rows.append(row)
    out = {"wandb_id": rid, "label": label, "suffix": sfx, "state": r.state,
           "last_step": r.lastHistoryStep, "rows": rows}
    with open(f"wandb_{label}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{label}: {len(rows)} eval points, state={r.state}, last_step={r.lastHistoryStep}")
    if rows:
        last = rows[-1]
        print("  final @ step", last.get("_step"))
        for k in K:
            if k in last:
                print(f"    {k.split('/')[-2][:22]:24s} {k.split('/')[-1]:42s} {last[k]:.4f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
