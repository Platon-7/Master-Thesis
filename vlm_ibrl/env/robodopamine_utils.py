"""RoboDopamine GRM scorer — env-agnostic, reusable across MetaWorld / Robomimic.

Faithful port of the authors' inference (github.com/FlagOpen/Robo-Dopamine,
`examples/inference.py`) to a transformers-based, per-call scorer usable inside an
RL loop. The GRM is a Qwen3-VL that, given a task instruction plus:
    - REFERENCE START (front image, task just starting),
    - REFERENCE END   (front image, goal / task completed),
    - a BEFORE three-view set (front, left-wrist, right-wrist),
    - an AFTER  three-view set (front, left-wrist, right-wrist),
outputs a *relative progress hop* as an integer percentage:
    <score>+NN%</score> | <score>-NN%</score> | <score>0%</score>   in [-100%, +100%].

Eval modes (how the caller picks BEFORE, and how a raw hop becomes progress):
    - "forward":     BEFORE = episode start   -> raw output IS absolute progress start->current
    - "incremental": BEFORE = previous step    -> raw output IS the per-step hop
    - "backward":    BEFORE = goal (x3 views)  -> progress = 1 + raw output
Authors default to "forward"; we follow that.

Single-camera envs: repeat the one view across all three (the authors' documented
single-view mode). For a genuine multi-view env, pass three distinct views.

Design note for the Robomimic port: this module knows nothing about MetaWorld. To
reuse it, an env only needs to (1) supply 3-view PIL frames per state, (2) supply a
front goal image, and (3) drive `RoboDopamineScorer.score(...)` + `accumulate_progress`.
"""
from __future__ import annotations

import re
import warnings
from typing import List, Optional

import torch
from PIL import Image

# Verbatim from the authors' examples/inference.py (SYSTEM_PROMPT). The 8 `<image>`
# placeholders are, in order: Ref-Start, Ref-End, BEFORE front/left/right, AFTER
# front/left/right. Keep byte-for-byte identical to preserve the trained behaviour.
ROBODOPAMINE_SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START->END continuum.
2) Direction:
   - AFTER better than BEFORE -> positive score.
   - AFTER worse than BEFORE -> negative score.
   - Essentially the same -> 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged -> 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""

# Authors' default token budget for the two images (min/max pixels on the image processor).
DEFAULT_MIN_PIXELS = 12544
DEFAULT_MAX_PIXELS = 76800

# The model emits integer OR float percentages (e.g. "+30%", "0.0%"), so accept a decimal.
_SCORE_RE = re.compile(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%\s*</score>")


def get_robodopamine(model_path: str, min_pixels: int = DEFAULT_MIN_PIXELS,
                     max_pixels: int = DEFAULT_MAX_PIXELS):
    """Load a Robo-Dopamine GRM (Qwen3-VL based). Returns (model, processor).
    Prefers flash_attention_2; falls back to the default attention if unavailable."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
        )
    except (ImportError, ValueError) as e:
        warnings.warn(f"flash_attention_2 unavailable for {model_path} ({e}); "
                      "falling back to default attention.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
    return model, processor


def parse_hop(text: str) -> Optional[float]:
    """Parse `<score>±NN%</score>` -> float in [-1, 1]. Returns None if unparseable.
    Takes the LAST match (the model may reason before emitting the final score)."""
    if not text:
        return None
    matches = _SCORE_RE.findall(text)
    if matches:
        return max(-100.0, min(100.0, float(matches[-1]))) / 100.0
    # Fallback: the authors' split-based parse (handles odd spacing / missing tag close).
    if "<score>" in text:
        try:
            val_str = text.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
            return max(-100.0, min(100.0, float(val_str))) / 100.0
        except Exception:
            return None
    return None


def accumulate_progress(eval_mode: str, raw: float, prev_progress: float) -> float:
    """Convert a raw model output into an accumulated global progress in ~[0, 1],
    exactly matching the authors' post-processing for each eval mode."""
    if eval_mode == "forward":
        return raw                                   # raw IS progress(start -> current)
    if eval_mode == "backward":
        return 1.0 + raw                             # raw is (current vs goal), usually <=0
    if eval_mode == "incremental":
        if raw >= 0:
            return prev_progress + (1.0 - prev_progress) * raw
        return prev_progress + prev_progress * raw
    raise ValueError(f"Unknown eval_mode: {eval_mode}")


class RoboDopamineScorer:
    """One GRM scorer bound to a task + fixed reference anchors for an episode.

    Usage per episode:
        scorer = RoboDopamineScorer(model, processor, task=..., goal_img=...,
                                    ref_start_img=..., eval_mode="forward")
        raw = scorer.score(before_views, after_views)   # before/after = [front, left, right]
        progress = accumulate_progress(scorer.eval_mode, raw, prev_progress)
    where each *_views is a list of 3 PIL.Images (repeat one view for single-camera envs).
    """

    def __init__(self, model, processor, *, task: str, goal_img: Image.Image,
                 ref_start_img: Image.Image, eval_mode: str = "forward",
                 max_new_tokens: int = 512):
        self.model = model
        self.processor = processor
        self.task = task
        self.goal_img = goal_img.convert("RGB")
        self.ref_start_img = ref_start_img.convert("RGB")
        self.eval_mode = eval_mode
        self.max_new_tokens = max_new_tokens
        # Pre-split the prompt once: 9 text segments around the 8 image slots.
        self._segments = ROBODOPAMINE_SYSTEM_PROMPT.format(task=task).split("<image>")
        assert len(self._segments) == 9, f"expected 8 <image> slots, got {len(self._segments)-1}"

    def _build_messages(self, before_views: List[Image.Image], after_views: List[Image.Image]):
        # The 8 images in the authors' fixed order.
        images = [self.ref_start_img, self.goal_img,
                  before_views[0], before_views[1], before_views[2],
                  after_views[0], after_views[1], after_views[2]]
        content = []
        for i, seg in enumerate(self._segments):
            content.append({"type": "text", "text": seg})
            if i < 8:
                content.append({"type": "image", "image": images[i]})
        return [{"role": "user", "content": content}], images

    @torch.inference_mode()
    def score(self, before_views: List[Image.Image], after_views: List[Image.Image],
              debug: bool = False) -> Optional[float]:
        """Return the raw hop in [-1, 1] (None if the model output can't be parsed)."""
        from qwen_vl_utils import process_vision_info
        messages, _ = self._build_messages(
            [im.convert("RGB") for im in before_views],
            [im.convert("RGB") for im in after_views],
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out = self.processor.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        if debug:
            print(f"[robodopamine] raw output: {out!r}", flush=True)
        return parse_hop(out)
