"""Decisive OOD-vs-fixable check for LRM-Progress on MetaWorld: can it tell an
OBVIOUS success (the clean goal image) from a start frame? If progress(goal) >>
progress(start), the model reads MetaWorld (our rollouts were just ambiguous). If
goal ~ start ~ 0.25, it's genuinely OOD. Also tries the initial-observation prompt
variant (some LRM progress prompts include an initial anchor)."""
import os, sys
os.environ.setdefault("V3_CORNER2_ZOOM", "1")
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.lrm_utils import get_lrm_progress, LRMProgressScorer, extract_progress_score
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR
from PIL import Image
import torch

RES = 240
GOALS = {"CoffeePush": "/shared/home/PKA4388/robodopamine_goals/coffeepush.png",
         "BoxClose":   "/shared/home/PKA4388/robodopamine_goals/boxclose.png"}


def start_frame(env_name):
    env = MetaWorldEnv(env_name, camera_name="corner2", width=RES, height=RES)
    env.reset()
    return Image.fromarray(env.render(camera_name="corner2_default", width=RES, height=RES))


def score_with_initial(model, processor, task, init_img, cur_img):
    """Prompt variant WITH an initial-observation anchor (authors' other branch)."""
    prompt = (f"Task: Estimate the completion progress.\nThe task is: {task}\nYou are given:"
              f"\n- Initial observation: <image>\n- Current observation: <image>\n\n"
              "Question: Based on the task description, estimate the completion progress as a value "
              "between 0.0 and 1.0.\nSelect one value from the following list:\n"
              "[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]\n\nDefinitions:\n"
              "- 0.0: Task has just started.\n- 0.1 - 0.9: Intermediate progress steps.\n"
              '- 1.0: Task is Finished.\nOutput your answer in the following JSON format:\n'
              '{ "completion_progress": selected_value }')
    segs = prompt.split("<image>")
    imgs = [init_img.convert("RGB"), cur_img.convert("RGB")]
    content = []
    for i, seg in enumerate(segs):
        if seg.strip():
            content.append({"type": "text", "text": seg})
        if i < 2:
            content.append({"type": "image", "image": imgs[i]})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    out = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return extract_progress_score(out)


def main():
    model, processor = get_lrm_progress()
    print("[lrm2] loaded", flush=True)
    for env_name, goalp in GOALS.items():
        task = TASK_STR[env_name]
        goal = Image.open(goalp).convert("RGB")
        start = start_frame(env_name)
        sc = LRMProgressScorer(model, processor, task=task)
        pg_single_goal = sc.score(goal)
        pg_single_start = sc.score(start)
        pg_init_goal = score_with_initial(model, processor, task, start, goal)
        pg_init_start = score_with_initial(model, processor, task, start, start)
        print(f"  [{env_name}] SINGLE-frame: progress(goal-success)={pg_single_goal:.2f}  "
              f"progress(start)={pg_single_start:.2f}  -> gap={pg_single_goal-pg_single_start:+.2f}", flush=True)
        print(f"  [{env_name}] WITH-initial: progress(goal)={pg_init_goal:.2f}  "
              f"progress(start)={pg_init_start:.2f}  -> gap={pg_init_goal-pg_init_start:+.2f}", flush=True)
    print("\n[lrm2] gap>>0 on the obvious goal => model reads MetaWorld (rollouts ambiguous); "
          "gap~0 => genuinely OOD.", flush=True)


if __name__ == "__main__":
    main()
