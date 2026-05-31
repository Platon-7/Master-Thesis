"""Compose an elegant 3-row figure for the score=4 inference probe.

Each row shows:
  - a horizontal strip of 6 representative frames from the episode
    (mostly early since the failures "freeze" partway through)
  - below the strip: side-by-side bar charts of model predictions
    (success_prob on the left, progress_reward on the right)

Models: 4B baseline, Robometer-FT run1 s3000, Qwen3.5-FT run4 s6500.
For each model: two bars — no-ICL (solid) and +ICL (hatched).

GT: terminal_reward = 4 (near-success failure). A well-calibrated model
on this curated label should output moderate success_prob and high
progress. Failure to match is the signal the user is looking for.

Hard-codes the prediction values from the diag job. Update PREDS when
the Qwen3.5-FT (run4 s6500) numbers land.
"""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image


DATA_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")
FRAME_PICK = [0, 2, 4, 7, 11, 15]  # 4 early + 1 mid + 1 last

EXAMPLES = [
    {
        "label": "MW-Assembly (held out)",
        "subtitle": "Pick up a nut and place it onto a peg — freeze failure",
        "family_dir": DATA_ROOT / "metaworld",
        "archive": "metaworld_assembly_v3",
        "episode_id": "metaworld_assembly_v3_score4_inst0000_corner2_freeze_score_4",
    },
    {
        "label": "MW-BinPicking (in training)",
        "subtitle": "Grasp the puck from one bin and place it into another bin — freeze failure",
        "family_dir": DATA_ROOT / "metaworld",
        "archive": "metaworld_bin_picking_v3",
        "episode_id": "metaworld_bin_picking_v3_score4_inst0000_corner2_freeze_score_4",
    },
    {
        "label": "FS-Pick (in training)",
        "subtitle": "Pick up the red cube and lift it to the goal — grasp-freeze failure",
        "family_dir": DATA_ROOT / "failsafe",
        "archive": "failsafe_pick",
        "episode_id": "failsafe_pick_pick_s4_grasp_freeze_inst0000_front_score_4",
    },
]

# (model_label, color)
MODELS = [
    ("Robometer-4B",       "#999999"),
    ("Robometer-FT s3000", "#1f6feb"),
    ("Qwen3.5-FT s6500",   "#ff6f3c"),
]

# Predictions:
#   Robometer-4B + Robometer-FT — from diag job 23037095 (demo2reward env, transformers 4.57)
#   Qwen3.5-FT — from diag job 23037297 (vlm_ibrl_qwen35 env + ROBOMETER_FORCE_FP32=1
#                to dodge cold-GPU bf16 NaN bug on asymmetric checkpoints)
PREDS: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "MW-Assembly (held out)": {
        "Robometer-4B":       {"no-ICL": (0.0566, 0.4790), "+ICL": (0.0889, 0.4620)},
        "Robometer-FT s3000": {"no-ICL": (0.4707, 1.0000), "+ICL": (0.2080, 0.8516)},
        "Qwen3.5-FT s6500":   {"no-ICL": (0.0010, 0.0000), "+ICL": (0.0042, 0.0000)},
    },
    "MW-BinPicking (in training)": {
        "Robometer-4B":       {"no-ICL": (0.0206, 0.0594), "+ICL": (0.0864, 0.4776)},
        "Robometer-FT s3000": {"no-ICL": (0.0693, 0.0000), "+ICL": (0.1367, 0.0000)},
        "Qwen3.5-FT s6500":   {"no-ICL": (0.0044, 0.0000), "+ICL": (0.0267, 1.0000)},
    },
    "FS-Pick (in training)": {
        "Robometer-4B":       {"no-ICL": (0.9102, 0.8875), "+ICL": (0.2812, 0.6251)},
        "Robometer-FT s3000": {"no-ICL": (0.2734, 1.0000), "+ICL": (0.5117, 1.0000)},
        "Qwen3.5-FT s6500":   {"no-ICL": (0.0005, 0.0000), "+ICL": (0.0005, 0.0000)},
    },
}

OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug/score4_diag")
OUT_DIR.mkdir(exist_ok=True)


def load_frames(family_dir: Path, archive: str, episode_id: str) -> Dict[int, np.ndarray]:
    """Return a dict {frame_idx: HxWx3 uint8} for an episode."""
    keyframes_dir = family_dir / "keyframes" / archive
    idx_path = keyframes_dir / "shard_index.json"
    idx = json.loads(idx_path.read_text())
    shard_path = keyframes_dir / idx[episode_id]
    out: Dict[int, np.ndarray] = {}
    prefix = episode_id + "/"
    with tarfile.open(shard_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.startswith(prefix) or not m.name.endswith(".jpg"):
                continue
            try:
                frame_idx = int(m.name[len(prefix):].split("_")[1])
            except (IndexError, ValueError):
                continue
            f = tf.extractfile(m)
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            out[frame_idx] = np.asarray(img)
    return out


def slugify(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def render_one(ex):
    """Render a single example as a standalone PNG: frame strip on top, two bar charts below."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n_frames = len(FRAME_PICK)
    fig = plt.figure(figsize=(13, 7.2))

    outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.05, 1.45],
        hspace=0.45,
        top=0.86, bottom=0.07, left=0.05, right=0.97,
    )
    frames_gs = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=outer[0], wspace=0.08)
    bars_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.22)

    # --- frame strip ---
    frames = load_frames(ex["family_dir"], ex["archive"], ex["episode_id"])
    for fi, fidx in enumerate(FRAME_PICK):
        ax = fig.add_subplot(frames_gs[0, fi])
        ax.imshow(frames[fidx])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999999"); s.set_linewidth(0.7)
        ax.set_title(f"frame {fidx}", fontsize=10, pad=4, color="#333333")

    # --- bar charts ---
    for metric_key, metric_title, ax_idx in [
        (0, "success_prob",   0),
        (1, "progress_reward", 1),
    ]:
        ax = fig.add_subplot(bars_gs[0, ax_idx])
        ax.set_ylim(0, 1.10)
        ax.set_title(metric_title, fontsize=13, color="#222222", pad=8)

        n_models = len(MODELS)
        xs = np.arange(n_models)
        width = 0.38

        preds_for_ex = PREDS[ex["label"]]
        for i, (mlabel, mcolor) in enumerate(MODELS):
            vals = preds_for_ex[mlabel]
            v_noicl = vals["no-ICL"][metric_key]
            v_icl   = vals["+ICL"][metric_key]

            if v_noicl is not None:
                ax.bar(xs[i] - width/2, v_noicl, width=width,
                       color=mcolor, edgecolor="#222222", linewidth=0.6)
                ax.text(xs[i] - width/2, v_noicl + 0.025, f"{v_noicl:.2f}",
                        ha="center", va="bottom", fontsize=10, color="#222222")
            else:
                ax.bar(xs[i] - width/2, 0.02, width=width, color="none",
                       edgecolor=mcolor, linewidth=1.0, alpha=0.5)
                ax.text(xs[i] - width/2, 0.05, "pending", ha="center",
                        va="bottom", fontsize=9, color="#aaaaaa", style="italic")

            if v_icl is not None:
                ax.bar(xs[i] + width/2, v_icl, width=width,
                       color=mcolor, edgecolor="#222222", linewidth=0.6,
                       hatch="///", alpha=0.85)
                ax.text(xs[i] + width/2, v_icl + 0.025, f"{v_icl:.2f}",
                        ha="center", va="bottom", fontsize=10, color="#222222")
            else:
                ax.bar(xs[i] + width/2, 0.02, width=width, color="none",
                       edgecolor=mcolor, linewidth=1.0, alpha=0.5, hatch="///")
                ax.text(xs[i] + width/2, 0.05, "pending", ha="center",
                        va="bottom", fontsize=9, color="#aaaaaa", style="italic")

        ax.set_xticks(xs)
        ax.set_xticklabels([m[0] for m in MODELS], fontsize=11)
        ax.tick_params(axis="x", pad=4)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if ax_idx == 0:
            ax.set_ylabel("probability", fontsize=11)

    # --- legend ---
    legend_handles = [
        Rectangle((0,0), 1, 1, facecolor="#666666", edgecolor="#222222", label="no-ICL"),
        Rectangle((0,0), 1, 1, facecolor="#666666", edgecolor="#222222", hatch="///", alpha=0.85, label="+ICL"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.97, 0.96), frameon=False, fontsize=11, ncol=2)

    fig.suptitle(
        f"{ex['label']}  —  GT: terminal_reward = 4 (near-success failure)\n"
        f"{ex['subtitle']}",
        fontsize=14, y=0.97, color="#111111",
    )

    out_png = OUT_DIR / f"score4_{slugify(ex['label'])}.png"
    out_pdf = OUT_DIR / f"score4_{slugify(ex['label'])}.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def main():
    print(f"writing 3 separate PNG/PDF pairs under {OUT_DIR}/")
    for ex in EXAMPLES:
        render_one(ex)
    print("done")


if __name__ == "__main__":
    main()
