"""Regenerate ECE + reliability diagram using the per-frame progress signal as P(success).

For all 3 models, we compute calibration on the per-frame progress prediction (P(success)
proxy that lives in [0, 1] for all heads):
- Loss 1 (CORN): progress_pred is already source-decoded scalar in [0, 1] = (1/4) * Σ σ(z_k).
  This is the closest thing to σ(z_{t,5}) we can recover from the dumped data.
- Loss 2 / Baseline (C51): progress_pred is [T, 10] logits → softmax × bin_centers → scalar [0, 1].

Calibration target: per-frame success_labels (whether the frame is in the "success" tail of
the trajectory).

This complements fig_7 (which uses the success head's sigmoid).
"""
import json, csv, os, numpy as np
import matplotlib.pyplot as plt

OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
PATHS = {
    "baseline": "/projects/prjs1958/LoRA_weights/test_eval/baseline/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss1":    "/projects/prjs1958/LoRA_weights/test_eval/loss1/robometer_lora_loss1_corn_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss2":    "/projects/prjs1958/LoRA_weights/test_eval/loss2/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
}
LABELS = {"baseline": "Baseline (Robometer-4B)",
          "loss1":    "Loss 1 (CORN, ours)",
          "loss2":    "Loss 2 (C51 + BCE asym)"}
COLORS = {"baseline": "#666666", "loss1": "#2E86AB", "loss2": "#E63946"}


def progress_signal(rec, model):
    pp = np.array(rec["progress_pred"])
    if model == "loss1":
        # CORN already source-decoded to [T] in [0, 1]
        return pp.flatten()
    # C51: [T, 10] logits -> softmax * bin_centers
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return (probs * np.linspace(0, 1, 10)).sum(axis=-1)
    return pp.flatten()


def per_frame(path, model):
    with open(path) as f:
        recs = json.load(f)
    seen = set(); probs, labels = [], []
    for r in recs:
        if r["id"] in seen: continue
        seen.add(r["id"])
        sig = progress_signal(r, model).flatten()
        sl = np.array(r["success_labels"]).flatten()
        n = min(len(sig), len(sl))
        probs.extend(sig[:n].tolist())
        labels.extend(sl[:n].astype(int).tolist())
    return np.array(probs), np.array(labels)


def compute_ece_bins(probs, labels, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_acc = np.zeros(n_bins); bin_conf = np.zeros(n_bins); bin_count = np.zeros(n_bins)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        cnt = mask.sum()
        if cnt == 0: continue
        bin_acc[i] = labels[mask].mean()
        bin_conf[i] = probs[mask].mean()
        bin_count[i] = cnt
        ece += (cnt / n) * abs(bin_acc[i] - bin_conf[i])
    return ece, bin_acc, bin_conf, bin_count, centers, edges


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    rows = []
    for idx, m in enumerate(["baseline", "loss1", "loss2"]):
        probs, labels = per_frame(PATHS[m], m)
        ece, bin_acc, bin_conf, bin_count, centers, edges = compute_ece_bins(probs, labels, n_bins=10)
        n = len(probs)
        rows.append([LABELS[m], ece, n, probs.min(), probs.max(),
                     float(np.median(probs)), int((probs > 0.1).sum()), int((probs > 0.5).sum())])
        print(f"{LABELS[m]:<28}  ECE = {ece:.4f}  (n_frames = {n}, range = [{probs.min():.3f}, {probs.max():.3f}])")

        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
        nonempty = bin_count > 0
        ax.bar(centers[nonempty], bin_acc[nonempty], width=0.09,
               alpha=0.7, color=COLORS[m], edgecolor="black", linewidth=0.5)
        # Bin counts on top of bars (so reader sees how many frames per bin)
        for c, cnt, acc in zip(centers[nonempty], bin_count[nonempty], bin_acc[nonempty]):
            ax.text(c, acc + 0.02, f"n={int(cnt)}", ha="center", fontsize=7, color="black")
        ax.set_xlabel("Predicted P(success)  [from per-frame progress signal]")
        if idx == 0:
            ax.set_ylabel("Empirical P(success)")
        ax.set_title(f"{LABELS[m]}\nECE = {ece:.4f}  (n={n})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Reliability diagrams — per-frame progress signal (CORN-decoded for L1, C51 expected-value for L2/baseline)\n"
                 "Replaces fig_7 calibration view that used the (frozen-for-L1) success head", fontsize=11)
    fig.tight_layout()
    out_path = f"{OUT}/fig_7b_reliability_progress_signal.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nWrote {out_path}")

    # CSV
    csv_path = f"{OUT}/table_8b_ece_progress_signal.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ECE", "n_frames", "pred_min", "pred_max", "pred_median",
                    "n_frames_above_0.1", "n_frames_above_0.5"])
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
