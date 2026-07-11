"""LRM (Large Reward Models, TRI/USC-PSI) — Absolute Progress Reward scorer.
Env-agnostic and reusable across MetaWorld / Robomimic.

Faithful port of the authors' `progress` mode (github.com/physical-superintelligence-lab/
Large-Reward-Models, vlm_reward/vlm_reward_server.py -> compute_vlm_reward reward_type=
"progress"). This is the model evaluated in the paper's Figure 2 (continuous progress
regression, `prog = LRM(I, d)`): a Qwen3-VL-8B that, from a SINGLE current frame + the
task, outputs a task-completion percentage in [0, 1] as JSON {"completion_progress": v}.

Purely zero-shot: no goal image, no demo, no multi-view (the single-frame form the paper
reports). The caller supplies one PIL frame per state; the reward is the progress value
(use it directly, or as a potential-based hop across steps — the RL env decides).

Robomimic port: nothing here is MetaWorld-specific. Feed Robomimic's current frame.
"""
from __future__ import annotations

import json
import re
import warnings
from typing import Optional

import torch
from PIL import Image

# Verbatim from the authors' server (reward_type == "progress"). Two zero-shot
# branches: with an INITIAL-observation anchor (their RL default caches one via
# `_update_initial_images_from_obs` "for progress-style prompts") and without.
# The initial anchor is what makes progress well-calibrated (empirically: without
# it the model returns a near-constant ~0.2 on MetaWorld; with it, start=0.0 and a
# success state scores clearly higher). The "initial" is the episode's own frame 0
# — NOT a demonstration — so this stays zero-shot. Keep wording identical.
LRM_PROGRESS_PROMPT = """Task: Estimate the completion progress.
The task is: {task}
You are given:
- Current observation: <image>

Question: Based on the task description, estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess how close to completion).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{{ "completion_progress": selected_value }}"""

LRM_PROGRESS_PROMPT_WITH_INITIAL = """Task: Estimate the completion progress.
The task is: {task}
You are given:
- Initial observation: <image>
- Current observation: <image>

Question: Based on the task description, estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess how close to completion).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{{ "completion_progress": selected_value }}"""

# HF repo hosts the three LRM models in subfolders; "progress" is the Figure-2 model.
DEFAULT_LRM_REPO = "USC-PSI-Lab/LRM-models"
DEFAULT_LRM_SUBFOLDER = "progress"


def get_lrm_progress(model_path: str = DEFAULT_LRM_REPO,
                     subfolder: str = DEFAULT_LRM_SUBFOLDER):
    """Load the LRM Absolute-Progress model (Qwen3-VL-8B). Returns (model, processor).
    Prefers flash_attention_2; falls back to default attention if unavailable."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    kw = dict(dtype=torch.bfloat16, device_map="auto")
    if subfolder:
        kw["subfolder"] = subfolder
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **kw)
    except (ImportError, ValueError) as e:
        warnings.warn(f"flash_attention_2 unavailable for {model_path} ({e}); "
                      "falling back to default attention.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kw)
    proc_kw = dict(trust_remote_code=True, use_fast=True)
    if subfolder:
        proc_kw["subfolder"] = subfolder
    processor = AutoProcessor.from_pretrained(model_path, **proc_kw)
    return model, processor


def extract_progress_score(response: str) -> float:
    """Parse the model output to a progress float in [0, 1] — the authors' exact
    cascade: JSON {"completion_progress": v} first, then regex fallbacks. 'NA' -> 0."""
    try:
        m = re.search(r"\{[^}]+\}", response)
        if m:
            data = json.loads(m.group(0))
            if "completion_progress" in data:
                v = data["completion_progress"]
                if str(v).upper() == "NA":
                    return 0.0
                return max(0.0, min(1.0, float(v)))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        m = re.search(r'"completion_progress"\s*:\s*"?(NA|([0-9.]+))"?', response, re.IGNORECASE)
        if m:
            cap = m.group(1)
            return 0.0 if cap.upper() == "NA" else max(0.0, min(1.0, float(cap)))
    except (ValueError, TypeError):
        pass
    for pat in (r"\b(0(\.[0-9]+)?|1(\.0)?)\b", r"[Pp]rogress[:\s]*([0-9]*\.?[0-9]+)",
                r"[Ss]core[:\s]*([0-9]*\.?[0-9]+)", r"([0-9]*\.?[0-9]+)\s*$", r"^([0-9]*\.?[0-9]+)"):
        m = re.search(pat, response)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                continue
    return 0.0


class LRMProgressScorer:
    """Single-frame LRM progress scorer bound to a task for an episode.

    Usage:
        # zero-shot with the episode's own frame-0 as the initial anchor (recommended):
        scorer = LRMProgressScorer(model, processor, task="Push a mug ...", initial_img=frame0)
        progress = scorer.score(current_frame_pil)   # float in [0, 1]
    initial_img may be None to fall back to the pure single-frame prompt.
    """

    def __init__(self, model, processor, *, task: str,
                 initial_img: Optional[Image.Image] = None, max_new_tokens: int = 128):
        self.model = model
        self.processor = processor
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.initial_img = initial_img.convert("RGB") if initial_img is not None else None
        tmpl = LRM_PROGRESS_PROMPT_WITH_INITIAL if self.initial_img is not None else LRM_PROGRESS_PROMPT
        self._prompt = tmpl.format(task=task)
        self._segments = self._prompt.split("<image>")
        self._n_images = len(self._segments) - 1   # 2 with initial, 1 without

    @torch.inference_mode()
    def score(self, current_frame: Image.Image, debug: bool = False) -> float:
        cur = current_frame.convert("RGB")
        images = [self.initial_img, cur] if self.initial_img is not None else [cur]
        content = []
        for i, seg in enumerate(self._segments):
            if seg.strip():
                content.append({"type": "text", "text": seg})
            if i < self._n_images:
                content.append({"type": "image", "image": images[i]})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out = self.processor.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        if debug:
            print(f"[lrm] raw output: {out!r}", flush=True)
        return extract_progress_score(out)
