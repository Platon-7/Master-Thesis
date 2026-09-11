"""Adapters that let the ManiSkill stack score the published VLM reward baselines.

Why this exists
---------------
The ManiSkill stack reaches its reward model through
`robometer.evals.eval_server.process_batch_helper`, which dispatches on
`model_type` and only knows the Robometer family ("robometer") and RoboReward
("roboreward"). RoboDopamine and LRM are different architectures with their own
loaders and prompts, so they cannot go through that path at all.

The MetaWorld/Robomimic stack (vlm_ibrl) already integrates both. Rather than
re-deriving their prompts and output parsing -- where a transcription slip would
silently produce a plausible-but-wrong reward -- `robodopamine_utils.py` and
`lrm_utils.py` are copied here verbatim, and this module is the thin adapter that
gives them the interface the ManiSkill buffer expects:

    score_clip(frames, task) -> (reward, success_prob)

matching `_compute_reward_single`'s contract. Neither baseline has a success
head, so success_prob is always 0.0; both are deployed as dense progress signals
(`reward_model.use_success_detection=false`), which is the form their authors
intend and the one the thesis uses for them.

Statelessness
-------------
`_compute_reward_single` is called once per clip with the episode's frames so
far, and holds no per-episode state. Both baselines fit that:

  * LRM scores the CURRENT frame anchored on the episode's frame 0, which is
    exactly frames[-1] anchored on frames[0].
  * RoboDopamine in "forward" eval mode returns progress(start -> current)
    directly (see accumulate_progress), so before=frames[0], after=frames[-1]
    needs no running total either.

RoboDopamine expects three camera views. ManiSkill renders one, so the view is
repeated three times -- the same fallback vlm_ibrl's robosuite env applies for
PickPlaceCan, which also has no second wrist camera.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from robometer.utils.logger import get_logger

logger = get_logger()

BASELINE_MODEL_TYPES = ("robodopamine", "lrm", "roboreward")

# RoboReward's own rubric prompt and answer format (vlm_ibrl/env/vlm_prompts.py).
ROBOREWARD_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.
"""


def _roboreward_to_reward(text: str) -> float:
    """`ANSWER: <1-5>` -> [0,1]. Unparseable output scores 0 rather than raising,
    so one malformed generation cannot kill a 300k-step run."""
    import re as _re
    m = _re.search(r"ANSWER:\s*([1-5])", text)
    if m:
        return (int(m.group(1)) - 1) / 4.0
    m = _re.search(r"\b([1-5])\b", text)
    if m:
        return (int(m.group(1)) - 1) / 4.0
    logger.warning(f"[roboreward] unparseable output {text[:80]!r} -> 0.0")
    return 0.0


def _to_pil(frame: Any) -> Image.Image:
    """Frames arrive as HWC uint8 arrays (or anything array-like)."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))           # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class BaselineRewardModel:
    """Common wrapper so the buffer can treat every baseline identically."""

    def __init__(self, model_type: str, model: Any, processor: Any, device: Any = None):
        self.model_type = model_type
        self.model = model
        self.processor = processor
        self.device = device if device is not None else next(model.parameters()).device
        self.max_frames = int(os.environ.get("BASELINE_MAX_FRAMES", "16"))
        # Per-episode anchors. The ManiSkill buffer calls us once per transition with
        # only the CURRENT frame (padded to max_frames by _pad_to_max_frames), so a
        # clip's first and last frames are the SAME image and a before/after model
        # sees no motion at all. vlm_ibrl avoids this by holding state on the env:
        # _rd_reset_episode stores start_views at reset and compares every later
        # frame against them; _lrm_reset_episode does the same for its initial
        # anchor. We reproduce that here, keyed on the episode id the buffer passes.
        self._ep_id = object()          # sentinel: first call always starts an episode
        self._ep_start: Optional[Image.Image] = None
        self._ep_frames: list = []
        self._goal_img: Optional[Image.Image] = None
        goal_path = os.environ.get("ROBODOPAMINE_GOAL", "")
        if model_type == "robodopamine":
            self.eval_mode = os.environ.get("ROBODOPAMINE_EVAL_MODE", "forward")
            if goal_path and os.path.exists(goal_path):
                self._goal_img = Image.open(goal_path).convert("RGB")
                logger.info(f"[robodopamine] goal anchor: {goal_path}")
            else:
                logger.warning(
                    f"[robodopamine] no goal image at ROBODOPAMINE_GOAL={goal_path!r}; "
                    "falling back to the clip's first frame (weak anchor)."
                )
        # Matched to the MetaWorld/Robomimic call sites: robosuite_vlm_env.py passes
        # max_new_tokens=24 for roboreward; robodopamine_utils/lrm_utils default to
        # 512/128 respectively.
        _default_tokens = {"robodopamine": "512", "roboreward": "24", "lrm": "128"}
        self.max_new_tokens = int(os.environ.get("BASELINE_MAX_NEW_TOKENS",
                                                 _default_tokens.get(model_type, "128")))

    def _cap_frames(self, frames: Any) -> Any:
        """Downsample a long clip to max_frames.

        The buffer PADS short clips up (_pad_to_max_frames) before calling us, but
        never reduces -- the Robometer path relies on raw_dict_to_sample to do that,
        and the baseline path skips it. ManiSkill episodes run 50 steps, so without
        this RoboReward would receive a 50-frame video where every other arm sees 16.
        Same linspace-with-repeats rule the buffer uses.
        """
        if frames is None:
            return frames
        n = len(frames)
        if n <= self.max_frames:
            return frames
        idx = np.linspace(0, n - 1, self.max_frames).round().astype(int)
        return [frames[i] for i in idx]

    # -- the interface the ManiSkill buffer calls -----------------------------
    def score_clip(self, frames: Any, task: str, episode_id: Any = None) -> Tuple[float, float]:
        _n_in = 0 if frames is None else len(frames)
        frames = self._cap_frames(frames)
        pil = [_to_pil(f) for f in frames]
        if not pil:
            return 0.0, 0.0

        # Current frame = the newest one the buffer holds.
        cur = pil[-1]
        if episode_id != self._ep_id:            # episode boundary -> re-anchor
            self._ep_id = episode_id
            self._ep_start = cur
            self._ep_frames = [cur]
        else:
            self._ep_frames.append(cur)
            if len(self._ep_frames) > self.max_frames:
                idx = np.linspace(0, len(self._ep_frames) - 1, self.max_frames).round().astype(int)
                self._ep_frames = [self._ep_frames[i] for i in idx]
        first, last = self._ep_start, cur
        self._dbg_calls = getattr(self, "_dbg_calls", 0) + 1
        if self._dbg_calls <= 8:
            same = bool(np.array_equal(np.asarray(first), np.asarray(last)))
            logger.info(f"[baseline-dbg] call={self._dbg_calls} ep={episode_id} "
                        f"n_in={_n_in} ep_frames={len(self._ep_frames)} start==cur={same}")
        if self.model_type == "lrm":
            # vlm_ibrl resizes to LRM_RES (256) before scoring; ManiSkill renders 224,
            # so resize here too rather than feeding the model an unfamiliar size.
            _res = int(os.environ.get("LRM_RES", "256"))
            if first.size != (_res, _res):
                first = first.resize((_res, _res))
            if last.size != (_res, _res):
                last = last.resize((_res, _res))
            from robometer_policy_learning.utils.lrm_utils import LRMProgressScorer
            scorer = LRMProgressScorer(self.model, self.processor, task=task,
                                       initial_img=first, max_new_tokens=self.max_new_tokens)
            return float(scorer.score(last)), 0.0
        if self.model_type == "roboreward":
            # RoboReward scores the WHOLE clip as a VIDEO against its 1-5 rubric
            # (use_video=is_roboreward in vlm_ibrl's robosuite env; the rubric text
            # itself says "the robot in the video"). Delegate the actual call to the
            # ported prompt_roboreward, which runs process_vision_info and passes
            # videos=/video_metadata= correctly -- building the processor call by
            # hand gives "Image features and image tokens do not match".
            from robometer_policy_learning.utils.roboreward_utils import prompt_roboreward
            messages = [
                {"role": "system", "content": [{"type": "text", "text": ROBOREWARD_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "video", "video": self._ep_frames, "sample_fps": 60,
                     "video_metadata": {"duration": len(self._ep_frames) / 60.0}},
                    {"type": "text", "text": f"{ROBOREWARD_PROMPT}\n\nTask: {task}"},
                ]},
            ]
            out = prompt_roboreward(
                model=self.model, processor=self.processor, messages=messages,
                prompt_kwargs=dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                                   top_p=1.0, top_k=0, temperature=0),
                debug=False,
            )
            return _roboreward_to_reward(out), 0.0

        if self.model_type == "robodopamine":
            from robometer_policy_learning.utils.robodopamine_utils import (
                RoboDopamineScorer, accumulate_progress,
            )
            scorer = RoboDopamineScorer(
                self.model, self.processor, task=task,
                goal_img=self._goal_img if self._goal_img is not None else first,
                ref_start_img=first, eval_mode=self.eval_mode,
                max_new_tokens=self.max_new_tokens,
            )
            raw = scorer.score([first] * 3, [last] * 3)
            if self._dbg_calls <= 8:
                logger.info(f"[baseline-dbg] robodopamine raw_hop={raw}")
            if raw is None:                       # unparseable output -> no signal
                logger.warning("[robodopamine] unparseable model output -> reward 0.0")
                return 0.0, 0.0
            return float(accumulate_progress(self.eval_mode, raw, 0.0)), 0.0
        raise ValueError(f"not a baseline model_type: {self.model_type}")


def _disable_cudnn_sdpa() -> None:
    """Force PyTorch off the cuDNN scaled-dot-product-attention kernel.

    The baseline loaders try flash_attention_2 and fall back to SDPA when it is
    unavailable. PyTorch then picks the cuDNN SDPA backend, which fails inside the
    reward model's forward with

        RuntimeError: Expected mha_graph.execute(...).is_good() to be true

    once the ManiSkill sim, the SAC networks and the reward model share one GPU.
    The same call succeeds standalone, so this is a backend-selection problem, not
    a shape problem. Disabling only the cuDNN kernel leaves flash / mem-efficient /
    math SDPA available, so throughput is essentially unchanged.
    """
    import torch
    for fn, val in (("enable_cudnn_sdp", False),):
        f = getattr(torch.backends.cuda, fn, None)
        if f is not None:
            try:
                f(val)
                logger.info(f"[baseline] torch.backends.cuda.{fn}({val})")
            except Exception as e:  # pragma: no cover - older torch
                logger.warning(f"[baseline] could not call {fn}: {e}")


def load_baseline_reward_model(model_type: str, model_path: Optional[str], device: Any):
    """Mirror of robometer.utils.save.load_model_from_hf for the two baselines.

    Returns (exp_cfg, tokenizer, processor, model) so the caller's unpacking is
    unchanged; exp_cfg is None and the "tokenizer" is the processor's own, which
    is all the ManiSkill path ever uses for these.
    """
    _disable_cudnn_sdpa()
    if model_type == "lrm":
        from robometer_policy_learning.utils.lrm_utils import (
            get_lrm_progress, DEFAULT_LRM_REPO, DEFAULT_LRM_SUBFOLDER,
        )
        repo = model_path or DEFAULT_LRM_REPO
        sub = os.environ.get("LRM_SUBFOLDER", DEFAULT_LRM_SUBFOLDER)
        model, processor = get_lrm_progress(repo, sub)
    elif model_type == "roboreward":
        from robometer_policy_learning.utils.roboreward_utils import _get_roboreward
        model, processor = _get_roboreward(model_path or "teetone/RoboReward-4B")
    elif model_type == "robodopamine":
        from robometer_policy_learning.utils.robodopamine_utils import get_robodopamine
        model, processor = get_robodopamine(
            model_path or "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview")
    else:
        raise ValueError(f"not a baseline model_type: {model_type}")
    wrapped = BaselineRewardModel(model_type, model, processor, device)
    logger.info(f"Loaded baseline reward model '{model_type}' from {model_path}")
    return None, getattr(processor, "tokenizer", None), processor, wrapped
