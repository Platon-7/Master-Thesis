"""Rebuild thesis Figure 5.10 from the coordinates extracted out of the original PDF."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("robomimic_rl.pdf.curves.json"))
# colours + markers read from the original PDF's own legend swatches
STYLE = {
    "RoboRef-ICL":       ((0.1216,0.4706,0.7059), "o"),
    "RoboRef-Std":       ((0.4157,0.2392,0.6039), "^"),
    "RoboDopamine":      ((0.8314,0.6275,0.0902), "v"),
    "LRM":               ((0.0,0.0,0.0),          "x"),
    "RoboRef-Asym":      ((0.8902,0.1020,0.1098), "s"),
    "Robometer-4B-Base": ((0.2000,0.6275,0.1725), "D"),
    "RoboReward":        ((0.5490,0.3373,0.2941), "+"),
}
ORDER = ["RoboRef-ICL","RoboRef-Std","RoboDopamine","LRM",
         "RoboRef-Asym","Robometer-4B-Base","RoboReward"]

fig, ax = plt.subplots(figsize=(11.2, 4.4), dpi=150)
ax.axhline(0.56, color="0.6", ls=(0,(3,3)), lw=1.2, zorder=1)
ax.text(750000, 0.575, "behavior cloning", ha="right", va="bottom",
        fontsize=9, color="0.45")
for name in ORDER:
    pts = d[f"PickPlace-Can|{name}"]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    c, m = STYLE[name]
    ax.plot(xs, ys, color=c, lw=2.0, marker=m, markersize=4.5,
            markevery=8, label=name, zorder=3)

ax.set_xlim(0, 760000); ax.set_ylim(-0.02, 0.90)
ax.set_xticks([0,100000,200000,300000,400000,500000,600000,700000])
ax.set_xticklabels(["0","100k","200k","300k","400k","500k","600k","700k"])
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_xlabel("Training steps", fontsize=12)
ax.set_ylabel("Task success rate (ground truth)", fontsize=12)
ax.set_title("Robomimic IBRL (PickPlace-Can): policy success by VLM reward model   (single seed)",
             fontsize=13, fontweight="bold")
ax.grid(True, color="0.9", lw=0.8)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.16), ncol=4,
          frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig("recreated.png", bbox_inches="tight")
print("wrote recreated.png")
