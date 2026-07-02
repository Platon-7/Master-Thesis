"""Visual check of what the reward model SEES.
Top 2 rows  = 16 ICL demo frames (icl_correct_view/<task>, 224, the success reference).
Bottom 2 rows = 16 live QUERY frames rendered from the env's corner2_default camera at
224 (the reward model's view) via an ORACLE rollout — built directly, no BC checkpoint.
If both halves share view/render/resolution, ICL is plugged in consistently."""
import os, sys, glob
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/shared/home/PKA4388/Master-Thesis/vlm_ibrl_v3")
os.environ["V3_CORNER2_ZOOM"] = "1"                 # match the RL runs
from env.metaworld_wrapper import MetaWorldEnv

TASK = os.environ.get("TASK", "coffeepush")
ENVNAME = {"coffeepush": "CoffeePush", "boxclose": "BoxClose"}[TASK]

env = MetaWorldEnv(ENVNAME, "corner2", 224, 224)
env.reset()
query = [env.render(camera_name="corner2_default", width=224, height=224)]
for t in range(160):
    a = env.get_heuristic_action()
    _, _, done, _ = env.step(a)
    query.append(env.render(camera_name="corner2_default", width=224, height=224))
    if done: break
print(f"[montage] {len(query)} query frames @ corner2_default 224", flush=True)
query = [Image.fromarray(q) for q in query]

frs = sorted(glob.glob(f"/shared/home/PKA4388/icl_correct_view/{TASK}/0_*.png"))
icl = [Image.open(frs[i]).convert("RGB") for i in np.linspace(0, len(frs)-1, 16).round().astype(int)]
q16 = [query[i] for i in np.linspace(0, len(query)-1, 16).round().astype(int)]

fig, axes = plt.subplots(4, 8, figsize=(16, 8.6))
imgs = icl + q16
for k, ax in enumerate(axes.flat):
    ax.imshow(imgs[k]); ax.axis("off")
axes.flat[0].set_title("ICL demo frames (16) — success reference shown to the model",
                       loc="left", fontsize=11, color="#1E6B2E")
axes.flat[16].set_title("live QUERY frames (16) — corner2_default, the model's actual view",
                        loc="left", fontsize=11, color="#1F3864")
fig.suptitle(f"What the reward model sees — {ENVNAME}   (top 2 rows ICL demos · bottom 2 rows live query · both 224x224)", fontsize=12)
fig.tight_layout()
out = f"/shared/home/PKA4388/Master-Thesis/reward-model-study/deck/icl_vs_query_{TASK}.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); print("[montage] saved", out, flush=True)
