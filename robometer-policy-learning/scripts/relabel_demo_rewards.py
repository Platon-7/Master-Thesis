#!/usr/bin/env python3
"""Replace the demos' ground-truth rewards with the VLM progress head's.

The demo h5 stores ManiSkill's normalized_dense GT reward, recorded at conversion.
That is fine while the demos are only an initialisation, but IBRL needs them IN the
replay buffer: the critic can only prefer the frozen BC action if it has been trained
on BC-like actions, and with sample_ratio=0 it never sees any -- so the BC proposal is
never selected and the floor IBRL is supposed to provide does not exist.

Putting the demos back in the buffer with GT rewards would leak ground truth into
half of every critic batch. Relabelling with the dense VLM reward gives the critic the
coverage it needs while keeping 100% of the learning signal from the reward model.
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-h5", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.utils.robometer_utils import extract_rewards_from_output
    from robometer_policy_learning.envs.maniskill_utils import get_task_spec

    spec = get_task_spec(args.task)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, proc, model = load_model_from_hf(model_path=args.model, device=dev)
    if not getattr(cfg.data, "use_multi_image", False):
        cfg.data.use_multi_image = True
    coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
    MF = int(getattr(cfg.data, "max_frames", 16))
    _plt = str(cfg.loss.progress_loss_type).lower()
    is_disc = _plt == "discrete" or "c51" in _plt
    nb = cfg.loss.progress_discrete_bins

    def score(frames):
        fr = np.stack(frames)
        if len(fr) < MF:
            fr = fr[np.linspace(0, len(fr) - 1, MF).round().astype(int)]
        raw = dict(frames=fr, task=spec.instruction, id=0,
                   metadata=dict(subsequence_length=len(fr)),
                   video_embeddings=None, text_embedding=None)
        s = raw_dict_to_sample(raw_data=raw, max_frames=MF, sample_type="progress")
        out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                                   batch_collator=coll, device=model.device, batch_data=[s.model_dump()],
                                   job_id=0, is_discrete_mode=is_disc, num_bins=nb)
        return float(extract_rewards_from_output(out)[0])

    with h5py.File(args.demo_h5, "r+") as f:
        d = f["data"]; keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        print(f"relabelling {len(keys)} demos in {os.path.basename(args.demo_h5)}", flush=True)
        for i, k in enumerate(keys):
            imgs = np.array(d[k]["obs"]["image"])
            r = np.array([score(list(imgs[: t + 1])) for t in range(len(imgs))], dtype=np.float32)
            d[k]["rewards"][...] = r
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(keys)}  last-ep reward mean={r.mean():.4f}", flush=True)
        f.attrs["rewards_source"] = "vlm_progress_head"
        f.attrs["reward_model"] = args.model
    print("DONE relabelling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
