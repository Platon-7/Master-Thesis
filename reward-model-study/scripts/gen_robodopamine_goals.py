"""Generate a front-view (corner2_default) success 'goal' image per task for
RoboDopamine — its required REFERENCE END anchor (the single-demo signal). Renders
an oracle success rollout and saves its final frame. Run once; the RL job points
ROBODOPAMINE_GOAL at the result. Rendered with the RL config (V3_CORNER2_ZOOM=1)
so the goal matches the reward feed."""
import os, sys
os.environ.setdefault("V3_CORNER2_ZOOM", "1")
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from PIL import Image

OUT = os.environ.get("ROBODOPAMINE_GOAL_DIR", "/shared/home/PKA4388/robodopamine_goals")
RES = int(os.environ.get("ROBODOPAMINE_RES", "240"))
FRONT_CAM = "corner2_default"
TASKS = {"coffeepush": "CoffeePush", "boxclose": "BoxClose"}
os.makedirs(OUT, exist_ok=True)


def oracle_success_final(env_name, max_tries=40):
    env = MetaWorldEnv(env_name, camera_name="corner2", width=RES, height=RES)
    for _ in range(max_tries):
        env.reset(); final = None; succ = 0
        for t in range(160):
            _, _, done, info = env.step(env.get_heuristic_action())
            if int(info.get("success", 0)) == 1:
                succ = 1
                final = env.render(camera_name=FRONT_CAM, width=RES, height=RES)
            if done:
                break
        if succ and final is not None:
            return final
    return None


def main():
    for tag, env_name in TASKS.items():
        img = oracle_success_final(env_name)
        if img is None:
            print(f"[goal] {tag}: FAILED to get an oracle success", flush=True); continue
        p = os.path.join(OUT, f"{tag}.png")
        Image.fromarray(img).save(p)
        print(f"[goal] {tag}: saved {p} ({img.shape})", flush=True)


if __name__ == "__main__":
    main()
