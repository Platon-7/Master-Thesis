import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from env.gvl import reward_from_GVL
from env.metaworld_wrapper import PixelMetaWorld
from env.qwen_utils import get_qwen3, get_qwen3_8b, prompt_qwen
from env.roboreward_utils import get_roboreward_4b, get_roboreward_8b, prompt_roboreward
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import (
    METAWORLD_DEMO2REWARD_REPLIES as demo2reward_replies,
    METAWORLD_TASK_DESCRIPTIONS as task_description,
    roboreward_prompt,
    vlmcritic_prompt,
)


# ---------------------------------------------------------------------------
# Per-task constants
# ---------------------------------------------------------------------------

# End-of-task time index for the first demo (0-indexed)
video_demo_t_end = {
    "Assembly": 49,
    "BoxClose": 68,
    "CoffeePush": 28,
    "StickPull": 52,
}

# End-of-task time indices for the first three demos
video_demo3_t_end = {
    "Assembly": [49, 43, 46],
    "BoxClose": [68, 55, 57],
    "CoffeePush": [28, 31, 30],
    "StickPull": [52, 54, 53],
}


# Valid VLM names accepted by VLMCritic_PixelMetaWorld
VALID_VLMS = (
    "vlm_sd_qwen3_8b",
    "vlm_sd_qwen3_32b",
    "demo2reward_qwen3_8b",
    "demo2reward_qwen3_32b",
    "roboreward_8b",
    "roboreward_4b",
    "robodopamine_4b",
    "lrm_progress_8b",
    "robometer_4b",
    "robometer_ft",   # user's fine-tuned variant; checkpoint via ROBOMETER_FT_PATH env var
    "qwen35_ft",      # alternative FT (Qwen3.5 base); checkpoint via QWEN35_FT_PATH env var
    "gvl_qwen3_32b",
    "gvl_qwen3_8b",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def past_frames_single_video(t, num_frames):
    if num_frames == 0:
        return []
    N_available = min(t + 1, num_frames)
    step = t / num_frames
    return [round(k * step) for k in range(N_available)]


def extract_answer_after_substring(text: str, substring="Final Instruction:"):
    s = text.strip()
    idx = s.rfind(substring)
    if idx == -1:
        first = next((ln for ln in s.splitlines() if ln.strip()), "")
        return first.strip()
    return s[idx + len(substring):].strip()


def single_prompt_eval(
    prompt_vlm,
    model,
    processor,
    system_prompt,
    prompt,
    frames,
    debug,
    use_video=False,
    max_new_tokens=5,
):
    start = time.perf_counter()
    content = []
    if use_video:
        content.append({
            "type": "video",
            "video": frames,
            "sample_fps": 20,
            "video_metadata": {"duration": len(frames) / 20.0},
        })
    else:
        for frame in frames:
            content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    precise_prompt = dict(max_new_tokens=max_new_tokens, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw_output = prompt_vlm(model=model, processor=processor, messages=messages,
                            debug=debug, prompt_kwargs=precise_prompt)
    if debug:
        elapsed_seconds = time.perf_counter() - start
        print(f".......... Elapsed time: {elapsed_seconds:.3f} .................")
    return raw_output


def response_to_reward(text: str) -> float:
    # Anything other than a single character "0"/"1" is treated as no success.
    if len(text.strip()) != 1:
        return 0.0
    return float(text.strip()[0])


def roboreward_to_reward(text: str) -> float:
    score_str = extract_answer_after_substring(text, substring="ANSWER:")
    assert score_str in ["1", "2", "3", "4", "5"], f"Got wrong answer {score_str}"
    score = int(score_str)
    return (score - 1) / 4.0  # normalize to [0, 1]


# ---------------------------------------------------------------------------
# VLM-critic-wrapped MetaWorld environment
# ---------------------------------------------------------------------------

# Default location for the 224x224 demo dataset that GVL queries.
DEFAULT_METAWORLD_DATA_DIR = "release/data/metaworld"
DEFAULT_GVL_CONTEXT_LEN = 9
DEFAULT_GVL_PERCS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _vlm_demo_dataset_path(metaworld_data_dir: str, env_name: str) -> Path:
    """Path to the 224x224 demo dataset that GVL uses as in-context examples."""
    return Path(metaworld_data_dir) / f"{env_name}_frame_stack_1_224x224_modem" / "dataset.hdf5"


def _otsu_threshold(x, bins=64):
    """Otsu's 1-D threshold over scores in [0,1] + a VARIANCE-NORMALIZED separation.

    Returns (threshold, d) where d = (mean_high - mean_low) / sigma_within is the
    standardized separation (Cohen's-d style) at the optimal Otsu split — the
    cluster gap measured in units of the within-cluster spread. This is
    scale-invariant (works whether successes sit at 0.3 or 0.9) and variance-aware
    (tight, well-resolved clusters score high; overlapping ones score low even if
    their means are far apart) — unlike a raw absolute gap. A unimodal blob, which
    Otsu still splits, lands around d~2.6 regardless of where/how wide it is, while
    a genuinely bimodal stream gives d well above ~5; the caller fires only when
    d >= a cutoff, else stays silent (graceful failure). Demo-free, GT-free."""
    x = np.asarray(x, dtype=float)
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return 1.1, 0.0, 1.1, 0.0
    p /= s
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    s2 = np.cumsum(p * centers ** 2)        # second-moment cumsum (for within-var)
    mu_t = mu[-1]; s2_t = s2[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, 1.0), 0.0)
    k = int(np.argmax(sigma_b))
    w0 = omega[k]
    if w0 <= 0 or w0 >= 1:
        return float(centers[k]), 0.0, float(centers[k]), 0.0
    mu_low = mu[k] / w0
    mu_high = (mu_t - mu[k]) / (1.0 - w0)
    # within-class variances of each side at the split
    var_low = max(s2[k] / w0 - mu_low ** 2, 0.0)
    var_high = max((s2_t - s2[k]) / (1.0 - w0) - mu_high ** 2, 0.0)
    sigma_w = float(np.sqrt(max(w0 * var_low + (1.0 - w0) * var_high, 1e-12)))  # pooled
    sigma_high = float(np.sqrt(max(var_high, 1e-12)))
    d = float((mu_high - mu_low) / sigma_w)
    # Returns: (valley threshold, separation d, high-cluster mean, high-cluster std).
    # The caller can fire at the valley OR at the high cluster (mu_high - k*sigma_high),
    # the latter being the demo-free analogue of "look as good as the success mode".
    return float(centers[k]), d, float(mu_high), sigma_high


class VLMCritic_PixelMetaWorld(PixelMetaWorld):
    """MetaWorld env wrapped with a frozen VLM reward model.

    Accepted ``vlm`` values:
        - ``vlm_sd_qwen3_8b`` / ``vlm_sd_qwen3_32b``           : VLM-SD zero-shot baseline (generic task instruction)
        - ``demo2reward_qwen3_8b`` / ``demo2reward_qwen3_32b`` : Demo2Reward critic with optimized prompts
        - ``roboreward_8b``                                    : RoboReward baseline
        - ``robometer_4b``                                     : Robometer-4B baseline (Qwen3-VL-4B w/ progress + success heads)
        - ``gvl_qwen3_8b`` / ``gvl_qwen3_32b``                 : GVL baseline
    """

    def __init__(self, *args, **kwargs):
        vlm = kwargs.pop("vlm")
        self.vlm_name = vlm
        assert vlm in VALID_VLMS, f"VLM {vlm} not recognized. Valid: {VALID_VLMS}"

        self.past_len = kwargs.pop("past_len", 4)
        assert self.past_len > 0, "past_len must be > 0"

        # End-of-episode handling
        self.end_on_success = kwargs["end_on_success"]
        self.reward_at_truncation = kwargs.pop("reward_at_truncation", 0)

        # Reward composition for Robometer-style critics (which expose both a
        # progress head and a success head). Final reward = beta * progress
        # + (1 - beta) * success_prob. Default 0.0 = pure success_prob, i.e.
        # behavior identical to the original integration. Ignored for non-
        # Robometer VLMs.
        self.robometer_beta = float(kwargs.pop("robometer_beta", 0.0))
        # Optional binarization threshold: if > 0, the mixed reward is
        # rewritten to 1.0 if it exceeds the threshold, else 0.0. Designed
        # for FT models whose success_prob lives in a small range like
        # [0.001, 0.08] — a calibrated threshold turns the soft output into
        # a usable IBRL sparse-reward signal. 0.0 = no thresholding (default).
        self.robometer_threshold = float(kwargs.pop("robometer_threshold", 0.0))
        # Optional reward scale applied before thresholding. Use when FT
        # outputs live in a small range and we want to keep the reward
        # continuous (no threshold) but bring it into IBRL's expected
        # [0, 1]-ish range. Order of operations: mixed *= scale → optional
        # threshold. Default 1.0 = no scaling.
        self.robometer_reward_scale = float(kwargs.pop("robometer_reward_scale", 1.0))
        self._last_progress = 0.0
        self._last_success_prob = 0.0

        # FAIR autonomous-RL mode (Robometer App. E-2): success detection AND
        # episode termination come from the MODEL's success_prob crossing
        # SUCCESS_THRESHOLD — never from the simulator's GT success. Enabled via
        # env var so SLURM jobs stay generic. When on, the scorer is queried
        # EVERY step (needed for detection), and GT success is returned only as a
        # logged diagnostic — the separate eval_env still measures true GT
        # success, which is the honest performance metric. Assumes a success
        # head (Robometer family / qwen35_ft). The detection threshold is
        # SEPARATE from robometer_threshold (which only shapes the reward value),
        # so reward shaping (beta / threshold / reward_at_truncation) stays
        # orthogonal to the success detector.
        self.autonomous = os.environ.get("AUTONOMOUS_SUCCESS", "0") == "1"
        self.success_threshold = float(os.environ.get("ROBOMETER_SUCCESS_THRESHOLD", "0.6"))
        # Debounce: require N CONSECUTIVE steps with success_prob > threshold
        # before declaring success + terminating (Christian's suggestion). Filters
        # transient single-frame false-positive spikes that otherwise end episodes
        # at ~step 20 and let the policy reward-hack. 1 = no debounce (default).
        self.success_consecutive = int(os.environ.get("ROBOMETER_SUCCESS_CONSECUTIVE", "1"))
        self._consec_success = 0
        # Which head drives success DETECTION (independent of the reward, which
        # is the beta mix). "success" = success_prob (paper default); "progress"
        # = progress_reward (rises monotonically, so it may avoid the early
        # single-frame spikes that make the success head terminate episodes
        # prematurely). Env var so jobs stay generic.
        self.detect_head = os.environ.get("ROBOMETER_DETECT_HEAD", "success")
        # SINGLE-FRAME-TILE (default OFF): replicate the reward model's Robometer-
        # authors LIBERO usage — instead of feeding the growing trajectory buffer,
        # each scorer call gets ONLY the current frame, tiled to fill max_frames
        # ([frame]*N). The progress head then reads a per-STATE progress estimate
        # (temporal context removed), used as a dense per-step reward. 0 = off;
        # any value >0 turns it on and (unless it is >1, in which case it is the
        # literal tile count) uses the scorer's own max_frames as N.
        self.single_frame_tile = int(os.environ.get("ROBOMETER_SINGLE_FRAME_TILE", "0"))
        # Stability levers (both default OFF; see the autonomous branch in step()):
        # CONFIRM_K: a candidate fire must be confirmed by a majority of K extra
        #   scores on jittered frame subsamples (self-ensemble majority vote).
        # REWARD_SAMPLE: terminal reward emitted as Bernoulli(success_prob) instead
        #   of the raw value (output sampling — systematic FPs stop paying out
        #   deterministically).
        self.confirm_k = int(os.environ.get("ROBOMETER_CONFIRM_K", "0"))
        self.confirm_frac = float(os.environ.get("ROBOMETER_CONFIRM_FRAC", "0.85"))
        # unanimous confirm vote (all ballots agree) vs majority
        self.confirm_unanimous = os.environ.get("ROBOMETER_CONFIRM_UNANIMOUS", "0") == "1"
        # pixel-augmentation strength for confirm-vote ballots (0 = off, demo-free)
        self.confirm_augment = float(os.environ.get("ROBOMETER_CONFIRM_AUGMENT", "0.0"))
        # dual-head: also require progress head > this to fire (0 = off, demo-free)
        self.progress_confirm_thr = float(os.environ.get("ROBOMETER_PROGRESS_CONFIRM_THR", "0.0"))
        self.reward_sample = os.environ.get("ROBOMETER_REWARD_SAMPLE", "0") == "1"
        # Online levers (both default OFF; see the autonomous branch in step()):
        # DYNAMIC_THR: adaptive quantile threshold over recent on-policy scores
        #   (no per-task calibration); SUCCESS_THRESHOLD acts as the floor.
        # MIN_EP_FRAC: ignore fires before frac×(median demo length) steps.
        self.dynamic_thr = os.environ.get("ROBOMETER_DYNAMIC_THR", "0") == "1"
        self.dyn_q = float(os.environ.get("ROBOMETER_DYN_Q", "0.995"))
        self.dyn_margin = float(os.environ.get("ROBOMETER_DYN_MARGIN", "0.03"))
        self.dyn_window = int(os.environ.get("ROBOMETER_DYN_WINDOW", "1500"))
        self._dyn_scores = []
        # Online-calibration levers (default OFF), compared against offline-fixed thr:
        # OTSU (demo-free): bar = Otsu valley of the recent on-policy score stream,
        #   used only when the stream is clearly BIMODAL (sep >= otsu_sep); else stay
        #   silent (bar=1.1) -> graceful failure. No demos, no GT, no per-task tuning.
        # DEMO_ANCHOR (demo-based): bar = percentile of the frozen model's success_prob
        #   on demo success windows loaded from ROBOMETER_DEMO_ANCHOR_PATH. Per-task, automatic.
        self.otsu = os.environ.get("ROBOMETER_OTSU", "0") == "1"
        self.otsu_min = int(os.environ.get("ROBOMETER_OTSU_MIN", "100"))
        self.otsu_bins = int(os.environ.get("ROBOMETER_OTSU_BINS", "64"))
        # min VARIANCE-NORMALIZED separation d = (mu_hi-mu_lo)/sigma_within at the
        # Otsu split to count the stream as bimodal (fire). Scale-invariant +
        # variance-aware: a unimodal blob lands ~2.6, real bimodal >>5. Below the
        # cutoff -> unimodal -> stay silent (graceful failure).
        self.otsu_d = float(os.environ.get("ROBOMETER_OTSU_D", "3.5"))
        # Otsu operates on PER-EPISODE PEAK success_probs, NOT the per-step stream.
        # The per-step stream is ~99% ordinary (failure-looking) frames, so its
        # "high cluster" is just the failure tail — genuine success frames are too
        # rare to form a cluster. Per-episode PEAKS separate failure-episodes (low
        # peak) from success-episodes (high peak), giving a real valley to fire at.
        self._ep_peak_scores = []          # rolling buffer of per-episode peak scores
        self._cur_ep_peak = 0.0            # running peak within the current episode
        self.otsu_min_eps = int(os.environ.get("ROBOMETER_OTSU_MIN_EPS", "20"))
        self.otsu_win_eps = int(os.environ.get("ROBOMETER_OTSU_WIN_EPS", "60"))
        self._otsu_next_log = self.otsu_min_eps   # diagnostic-print throttle (buffer-length based)
        self.demo_anchor = os.environ.get("ROBOMETER_DEMO_ANCHOR", "0") == "1"
        self.demo_anchor_path = os.environ.get("ROBOMETER_DEMO_ANCHOR_PATH", "")
        self.demo_anchor_pct = float(os.environ.get("ROBOMETER_DEMO_ANCHOR_PCT", "25"))
        self.demo_anchor_k = int(os.environ.get("ROBOMETER_DEMO_ANCHOR_K", "3"))
        self._demo_anchor_thr = None
        _min_frac = float(os.environ.get("ROBOMETER_MIN_EP_FRAC", "0"))
        _env_name_early = args[0] if args else kwargs.get("env_name")
        if _min_frac > 0 and _env_name_early in video_demo3_t_end:
            _demo_med = sorted(video_demo3_t_end[_env_name_early])[1]
            self.min_ep_steps = int(_min_frac * _demo_med)
        else:
            self.min_ep_steps = int(os.environ.get("ROBOMETER_MIN_EP_STEPS", "0"))
        if self.dynamic_thr or self.min_ep_steps > 0:
            print(f"[online-levers] dynamic_thr={self.dynamic_thr} "
                  f"(q={self.dyn_q}, margin={self.dyn_margin}, window={self.dyn_window}, "
                  f"floor={self.success_threshold}) min_ep_steps={self.min_ep_steps}", flush=True)

        # ICL context for Robometer-family scorers. When ROBOMETER_ICL_DEMO_PATH
        # is set (a directory of `{demo_idx}_NNN.png` frames), pick N uniform
        # frames from the chosen demo and pass them to every scorer call as
        # the in-context demonstration. Matches training-time ICL recipe.
        self.icl_frames = None
        icl_path = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
        print(
            f"[ICL debug] env var ROBOMETER_ICL_DEMO_PATH="
            f"{icl_path!r}  vlm={vlm!r}  → will load ICL: "
            f"{bool(icl_path and 'robometer' in vlm)}",
            flush=True,
        )
        if icl_path and "robometer" in vlm:
            from pathlib import Path as _P
            from PIL import Image as _PIL
            icl_demo_idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
            icl_n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
            frames_dir = _P(icl_path)
            available = sorted(
                p for p in frames_dir.iterdir()
                if p.name.startswith(f"{icl_demo_idx}_") and p.suffix == ".png"
            )
            if not available:
                raise FileNotFoundError(
                    f"ROBOMETER_ICL_DEMO_PATH={icl_path} has no frames for "
                    f"demo {icl_demo_idx}"
                )
            picks = np.linspace(0, len(available) - 1, icl_n).round().astype(int)
            self.icl_frames = [
                np.asarray(_PIL.open(available[i]).convert("RGB"), dtype=np.uint8)
                for i in picks
            ]
            print(
                f"[ICL] loaded {icl_n} frames from demo {icl_demo_idx} of "
                f"{icl_path} (indices {picks.tolist()})"
            )

        # Configurable: where GVL looks up its in-context demos.
        self.metaworld_data_dir = kwargs.pop("metaworld_data_dir", DEFAULT_METAWORLD_DATA_DIR)
        # Number of in-context demo frames (matches paper's Suppl. A.2).
        self.gvl_context_len = kwargs.pop("gvl_context_len", DEFAULT_GVL_CONTEXT_LEN)
        # Completion-% labels paired with the in-context demo frames.
        self.gvl_percs = list(kwargs.pop("gvl_percs", DEFAULT_GVL_PERCS))

        env_name = args[0] if args else kwargs["env_name"]
        self.rl_camera = kwargs.get("rl_camera")
        # Optional SEPARATE camera for VLM reward scoring. The policy still observes
        # rl_camera; only the frames fed to the reward model come from reward_camera.
        # Used to feed the reward model a legible view (e.g. topview) instead of the
        # zoomed corner2 that the reward model cannot read.
        self.reward_camera = kwargs.pop("reward_camera", None) or self.rl_camera

        # Pre-load GVL in-context demos.
        self.all_demos = []
        if "gvl" in vlm:
            for i in range(3):
                demo_frames = self._load_demo_video(env_name, self.rl_camera, i)
                t_final = video_demo3_t_end[env_name][i]
                demo_idx = past_frames_single_video(t_final, self.gvl_context_len)
                final_frame = demo_frames[-1]
                demo_frames = [demo_frames[i] for i in demo_idx]
                demo_frames.append(final_frame)
                self.all_demos.append(demo_frames)

        super().__init__(*args, **kwargs)

        task = task_description[env_name]
        self.task_description = task

        # Build (model, prompt) for the chosen critic.
        # VLM-SD and Demo2Reward share the same prompt template; only the `instruction`
        # body differs (generic task description vs Demo2Reward-optimized prompt).
        if vlm in ("vlm_sd_qwen3_8b", "vlm_sd_qwen3_32b", "demo2reward_qwen3_8b", "demo2reward_qwen3_32b"):
            use_32b = vlm.endswith("_32b")
            self.vlm, self.processor = get_qwen3() if use_32b else get_qwen3_8b()
            self.prompt_vlm = prompt_qwen
            self.system_prompt = vlmcritic_prompt
            if vlm.startswith("vlm_sd"):
                instruction = task  # generic task description
            else:
                instruction = demo2reward_replies[env_name]  # optimized prompt
            self.prompt = (
                f"Here is a sequence of frames showing a robot policy attempting to solve a task. "
                f"I need your help determining whether the policy is successful.\n\n"
                f"Instruction: {instruction}\n"
                f"Output EXACTLY a single character, either 0 or 1, to denote task completion. "
                f"Use 1 if the task is completed; 0 otherwise. Use no other symbols or formatting."
            )
        elif vlm == "roboreward_8b":
            self.vlm, self.processor = get_roboreward_8b()
            self.prompt_vlm = prompt_roboreward
            self.system_prompt = roboreward_prompt
            self.prompt = f"{roboreward_prompt}\n\nTask: {task}"
        elif vlm == "roboreward_4b":
            self.vlm, self.processor = get_roboreward_4b()
            self.prompt_vlm = prompt_roboreward
            self.system_prompt = roboreward_prompt
            self.prompt = f"{roboreward_prompt}\n\nTask: {task}"
        elif vlm == "robodopamine_4b":
            # Robo-Dopamine GRM: generative, MULTI-VIEW, hop-based progress. The model
            # loads here; the per-episode scorer (bound to a goal + reference-start
            # image) is built in reset(). Dense potential-based reward, scored every
            # ROBODOPAMINE_STRIDE steps. See env/robodopamine_utils.py. The whole
            # RoboDopamine path lives in the `is_robodopamine` branches of reset()/step()
            # so no other VLM's logic is affected.
            from env.robodopamine_utils import get_robodopamine
            _rd_path = os.environ.get("ROBODOPAMINE_PATH",
                                      "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview")
            self.vlm, self.processor = get_robodopamine(_rd_path)
            self._rd_cams = os.environ.get(
                "ROBODOPAMINE_CAMS", "corner2_default,gripperPOV,behindGripper").split(",")
            self._rd_stride = int(os.environ.get("ROBODOPAMINE_STRIDE", "16"))
            self._rd_eval_mode = os.environ.get("ROBODOPAMINE_EVAL_MODE", "forward")
            self._rd_goal_path = os.environ.get("ROBODOPAMINE_GOAL", "")
            # Render the reward views at a fixed, decent resolution (not the policy's
            # small rl_image_size) so the GRM gets clear input.
            self._rd_res = int(os.environ.get("ROBODOPAMINE_RES", "240"))
            self._rd_task = task
        elif vlm == "lrm_progress_8b":
            # LRM (Large Reward Models, TRI/USC-PSI) Absolute Progress Reward, the
            # Figure-2 model. Single-frame Qwen3-VL-8B -> completion progress [0,1].
            # Faithful to their config/progress.yaml downstream-RL recipe: an INITIAL
            # anchor (frame 0), queried every LRM_CALL_INTERVAL steps, and the reward
            # HELD (absolute progress) between queries (vlm_non_call_reward_mode: hold,
            # vlm_pure_reward). The per-episode scorer is built in reset(). Its whole
            # path lives in the `lrm_progress_8b` branches of reset()/step().
            from env.lrm_utils import get_lrm_progress
            self.vlm, self.processor = get_lrm_progress()
            self._lrm_interval = int(os.environ.get("LRM_CALL_INTERVAL", "10"))
            self._lrm_res = int(os.environ.get("LRM_RES", "256"))   # >= LRM's 256^2 min_pixels
            # Reward form. "hold" = absolute progress held between calls (their PPO
            # config/progress.yaml). "delta" = potential-based progress increment
            # (IBRL-appropriate; matches how RoboDopamine's dense reward is delivered
            # here). Their absolute-held reward acts as a survival bonus under IBRL's
            # off-policy Q-learning; delta gives a properly-shaped gradient.
            self._lrm_reward_mode = os.environ.get("LRM_REWARD_MODE", "hold")
            self._lrm_cam = os.environ.get("LRM_CAM", "corner2_default")
            self._lrm_use_initial = os.environ.get("LRM_INCLUDE_INITIAL", "1") == "1"
            self._lrm_task = task
        elif vlm in ("robometer_4b", "robometer_ft", "qwen35_ft"):
            # Robometer-family critics (progress + success heads, not
            # generative). All variants share one loader; only the
            # checkpoint path differs. FT variants read their path from an
            # env var so the SLURM job stays generic.
            _ckpt_map = {
                "robometer_4b": os.environ.get("ROBOMETER_4B_PATH", "robometer/Robometer-4B"),
                "robometer_ft": os.environ.get("ROBOMETER_FT_PATH", ""),
                "qwen35_ft":    os.environ.get("QWEN35_FT_PATH", ""),
            }
            ckpt = _ckpt_map[vlm]
            if not ckpt:
                raise ValueError(
                    f"--vlm {vlm} requires the matching env var (ROBOMETER_FT_PATH "
                    f"or QWEN35_FT_PATH) pointing at a consolidated checkpoint dir."
                )
            self.scorer = get_robometer_4b(model_path=ckpt)
            self.vlm = self.scorer
            self.processor = None
            self.prompt_vlm = None
            self.system_prompt = None
            self.prompt = None
            # Interval-hop shaping for the Robometer family (mirrors the _rd/_lrm
            # delivery): RBM_CALL_INTERVAL>0 activates _rbm_step_reward -- score the
            # last RBM_LASTK frames every interval steps, decode condMean from the
            # C51 bins, pay the potential hop prog_t - prog_{t-interval}; at
            # truncation optionally add the sparse success-head anchor (the winning
            # money-plot reward) when RBM_ANCHOR=1. 0 = disabled (default), all
            # existing behavior untouched.
            self._rbm_interval = int(os.environ.get("RBM_CALL_INTERVAL", "0") or 0)
            self._rbm_lastk = int(os.environ.get("RBM_LASTK", "4") or 4)
            self._rbm_anchor = os.environ.get("RBM_ANCHOR", "1") == "1"
            self._rbm_weight = float(os.environ.get("RBM_DELTA_WEIGHT", "1") or 1)
            self._rbm_log_n = 0
        elif vlm in ("gvl_qwen3_8b", "gvl_qwen3_32b"):
            self.vlm, self.processor = get_qwen3_8b() if vlm == "gvl_qwen3_8b" else get_qwen3()
            self.prompt_vlm = prompt_qwen
            # GVL builds its own prompt at call time.
            self.system_prompt = None
            self.prompt = None
        else:
            raise AssertionError(f"VLM {vlm} not recognized")  # unreachable

        self.vlm.eval()
        self.current_video = []
        self.vid_t = 0

        # DEMO-ANCHORED online calibration (demo-based): set the detection bar from
        # the frozen model's success_prob on a few demo success windows. Computed once
        # (model is frozen); no offline grid-search, no GT. Reuses the demo frame dir
        # ({demo_idx}_NNN.png) — same source/view as ICL.
        if self.demo_anchor and self.demo_anchor_path and (
            "robometer" in vlm or vlm == "qwen35_ft"
        ):
            from pathlib import Path as _P
            from PIL import Image as _PIL
            d = _P(self.demo_anchor_path)
            present = sorted({p.name.split("_")[0] for p in d.iterdir()
                              if p.suffix == ".png" and p.name.split("_")[0].isdigit()},
                             key=lambda s: int(s))
            scores = []
            for di in present[: self.demo_anchor_k]:
                frs = sorted(p for p in d.iterdir()
                             if p.name.startswith(f"{di}_") and p.suffix == ".png")
                if not frs:
                    continue
                picks = np.linspace(0, len(frs) - 1, 16).round().astype(int)
                # Feed PIL Images (same type the on-policy path and desc_rerank use);
                # passing numpy arrays here makes the scorer return garbage ~0.
                win = [_PIL.open(frs[i]).convert("RGB") for i in picks]
                out = self.scorer(win, task=self.task_description, icl_frames=self.icl_frames)
                scores.append(float(out["success_prob"]))
            if scores:
                self._demo_anchor_thr = float(np.percentile(scores, self.demo_anchor_pct))
                print(f"[demo-anchor] {len(scores)} demos success_probs="
                      f"{[round(s, 3) for s in scores]} -> bar(p{self.demo_anchor_pct:g})="
                      f"{self._demo_anchor_thr:.3f}", flush=True)
            else:
                print(f"[demo-anchor] WARNING: no demo frames found at {self.demo_anchor_path}", flush=True)

    def _load_demo_video(self, env_name, camera_name, i=0):
        data_path = _vlm_demo_dataset_path(self.metaworld_data_dir, env_name)
        if not data_path.exists():
            raise FileNotFoundError(
                f"VLM demo dataset not found at {data_path}. Pass metaworld_data_dir to "
                f"VLMCritic_PixelMetaWorld or place the 224x224 modem dataset at this path."
            )
        end_t = video_demo3_t_end[env_name][i]
        demo_frames = []
        with h5py.File(data_path, "r") as f:
            for t in range(end_t):
                img = Image.fromarray(
                    f[f"data/demo_{i}/obs/{camera_name}_image"][t].astype(np.uint8).transpose(1, 2, 0)
                )
                demo_frames.append(img)
        return demo_frames

    def fetch_img(self, obs):
        img_np = obs[self.reward_camera].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
        assert img_np.shape[0] == 224 and img_np.shape[1] == 224 and img_np.shape[2] == 3
        return Image.fromarray(img_np)

    def _augment_frame(self, img):
        """Small random photometric perturbation of a PIL frame for the
        augmentation-ensemble confirm vote. Scale = self.confirm_augment.
        Brightness/contrast jitter + light gaussian noise — enough to flip a
        pixel-brittle false positive, not enough to break a genuine success."""
        from PIL import ImageEnhance
        a = self.confirm_augment
        out = img
        out = ImageEnhance.Brightness(out).enhance(1.0 + random.uniform(-a, a))
        out = ImageEnhance.Contrast(out).enhance(1.0 + random.uniform(-a, a))
        arr = np.asarray(out, dtype=np.float32)
        arr = arr + np.random.normal(0.0, 255.0 * a * 0.5, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ---- Robo-Dopamine (multi-view, dense, hop-based) reward path --------------
    def _base_env(self):
        """Return the MetaWorldEnv in the wrapper chain — its render() accepts
        camera_name. Stop there; descending further reaches the raw gymnasium
        MujocoEnv whose render() has no camera_name arg. (Robomimic port: replace
        this check with whatever env exposes a per-camera render.)"""
        e = self.env
        while True:
            if type(e).__name__ == "MetaWorldEnv" or hasattr(e, "_reward_cam_name"):
                return e
            if not hasattr(e, "env"):
                return e
            e = e.env

    def _rd_render_views(self):
        """Render the current state from the RoboDopamine multi-view cameras
        (front + two wrist views) as PIL RGB. On-demand, so it costs nothing for
        the (majority of) steps we don't score."""
        base = self._base_env()
        R = self._rd_res
        return [Image.fromarray(base.render(camera_name=c, width=R, height=R))
                for c in self._rd_cams]

    def _rd_reset_episode(self):
        """Build the per-episode GRM scorer: bind it to this run's reference-start
        frame and the (task) goal image, and reset the accumulated-progress state."""
        from env.robodopamine_utils import RoboDopamineScorer
        start_views = self._rd_render_views()
        if self._rd_goal_path and os.path.exists(self._rd_goal_path):
            goal_img = Image.open(self._rd_goal_path).convert("RGB")
        else:
            # No goal supplied -> degenerate (start frame). Warn once; RL should set one.
            if not getattr(self, "_rd_goal_warned", False):
                print(f"[robodopamine] WARNING: no goal image at ROBODOPAMINE_GOAL="
                      f"{self._rd_goal_path!r}; using start frame (weak).", flush=True)
                self._rd_goal_warned = True
            goal_img = start_views[0]
        self._rd_scorer = RoboDopamineScorer(
            self.vlm, self.processor, task=self._rd_task, goal_img=goal_img,
            ref_start_img=start_views[0], eval_mode=self._rd_eval_mode)
        self._rd_start_views = start_views
        self._rd_prev_views = start_views
        self._rd_prev_prog = 0.0
        self._rd_step_i = 0

    def _rd_step_reward(self, rl_obs, success, image_obs):
        """Score every ROBODOPAMINE_STRIDE steps (and at the final step); deliver a
        potential-based hop reward (progress(s_t) - progress(s_{t-stride})). This
        telescopes to the final progress, is policy-invariant (the authors' shaping),
        and keeps the GRM call count low. Non-autonomous: episodes run to the timeout."""
        from env.robodopamine_utils import accumulate_progress
        self._rd_step_i += 1
        time_truncated = bool(self.most_recent_info["truncated"])
        reward = 0.0
        if self._rd_step_i % self._rd_stride == 0 or time_truncated:
            after_views = self._rd_render_views()
            before = (self._rd_start_views if self._rd_eval_mode == "forward"
                      else self._rd_prev_views)
            raw = self._rd_scorer.score(before, after_views)
            if raw is not None:
                prog = accumulate_progress(self._rd_eval_mode, raw, self._rd_prev_prog)
                reward = prog - self._rd_prev_prog          # potential-based hop
                self._rd_prev_prog = prog
                self.episode_stats["vlm_robometer_progress_sum"] = prog  # log final progress
            self._rd_prev_views = after_views
        self.episode_stats["vlm_reward_counts"] += 1
        # Non-autonomous: terminal only on the episode-length timeout (no GT leak).
        return rl_obs, float(reward), bool(time_truncated), success, image_obs

    # ---- LRM (Absolute Progress Reward) path -----------------------------------
    def _lrm_render(self):
        base = self._base_env()
        return Image.fromarray(base.render(camera_name=self._lrm_cam,
                                           width=self._lrm_res, height=self._lrm_res))

    def _lrm_reset_episode(self):
        """Bind the LRM scorer to this episode's frame-0 initial anchor (zero-shot)."""
        from env.lrm_utils import LRMProgressScorer
        init_img = self._lrm_render() if self._lrm_use_initial else None
        self._lrm_scorer = LRMProgressScorer(self.vlm, self.processor,
                                             task=self._lrm_task, initial_img=init_img)
        self._lrm_held = 0.0
        self._lrm_prev_prog = 0.0
        self._lrm_step_i = 0

    def _lrm_step_reward(self, rl_obs, success, image_obs):
        """Query the single-frame progress every LRM_CALL_INTERVAL steps.
        LRM_REWARD_MODE:
          "hold"  = absolute progress held between calls (their PPO config/progress.yaml).
          "delta" = potential-based progress increment (progress_t - progress_{t-interval}),
                    0 between calls — the IBRL-appropriate form (same delivery as
                    RoboDopamine's dense reward). Non-autonomous: runs to the timeout."""
        self._lrm_step_i += 1
        time_truncated = bool(self.most_recent_info["truncated"])
        reward = self._lrm_held  # "hold" default carries the last value between calls
        if self._lrm_step_i % self._lrm_interval == 0 or time_truncated:
            prog = max(0.0, min(1.0, float(self._lrm_scorer.score(self._lrm_render()))))
            if self._lrm_reward_mode == "delta":
                reward = prog - self._lrm_prev_prog
                self._lrm_prev_prog = prog
            else:
                self._lrm_held = prog
                reward = prog
            self.episode_stats["vlm_robometer_progress_sum"] = prog
        elif self._lrm_reward_mode == "delta":
            reward = 0.0  # delta pays out only at scored steps
        self.episode_stats["vlm_reward_counts"] += 1
        return rl_obs, float(reward), bool(time_truncated), success, image_obs

    def _rbm_hop_term(self):
        """Additive interval-hop shaping term (RBM_MODE=add, the default). Called at
        the step() return sites so the stock reward path, including the autonomous
        success protocol, runs untouched; on every RBM_CALL_INTERVAL-th step this
        adds w*(condMean_lastK(t) - condMean_lastK(t-interval)), 0 between. Returns
        0.0 when shaping is off or the VLM is not Robometer-family."""
        if not getattr(self, "_rbm_interval", 0) or os.environ.get("RBM_MODE", "add") == "replace":
            return 0.0
        if not ("robometer" in self.vlm_name or self.vlm_name == "qwen35_ft"):
            return 0.0
        self._rbm_step_i += 1
        if self._rbm_step_i % self._rbm_interval != 0 and not bool(self.most_recent_info["truncated"]):
            return 0.0
        out = self.scorer(self.current_video[-self._rbm_lastk:], task=self.task_description)
        bins = out.get("prog_bins_final")
        if bins:
            _b = np.asarray(bins, dtype=float)
            _c = np.linspace(0.0, 1.0, _b.shape[0])
            _ev = float((_b * _c).sum())
            _nz = max(1.0 - float(_b[0]), 1e-6)
            prog = float(min(_ev / _nz, 1.0)) if _nz > 0.05 else _ev
        else:
            prog = float(out["progress_reward"])
        hop = self._rbm_weight * (prog - self._rbm_prev_prog)
        self._rbm_prev_prog = prog
        if self._rbm_log_n < 12 or self._rbm_log_n % 500 == 0:
            print(f"[RBM-HOP] add i={self._rbm_step_i} prog={prog:.4f} hop={hop:+.4f}", flush=True)
        self._rbm_log_n += 1
        return hop

    def _rbm_step_reward(self, rl_obs, success, image_obs):
        """Interval-scored potential hops from the Robometer progress head, condMean
        decode over the last RBM_LASTK frames, optionally stacked on the sparse
        success-head anchor at truncation (RBM_ANCHOR=1). Same delivery as _rd/_lrm:
        score every RBM_CALL_INTERVAL steps (and at truncation), pay the hop
        prog_t - prog_{t-interval} at scored steps, 0 between; the hops telescope to
        the final progress and are policy-invariant. Non-autonomous: terminal only on
        the episode timeout. current_video holds PIL images (the scorer returns
        garbage on numpy arrays -- see the demo-anchor comment above)."""
        self._rbm_step_i += 1
        time_truncated = bool(self.most_recent_info["truncated"])
        reward = 0.0
        if self._rbm_step_i % self._rbm_interval == 0 or time_truncated:
            out = self.scorer(self.current_video[-self._rbm_lastk:], task=self.task_description)
            bins = out.get("prog_bins_final")
            if bins:
                _b = np.asarray(bins, dtype=float)
                _c = np.linspace(0.0, 1.0, _b.shape[0])
                _ev = float((_b * _c).sum())
                _nz = max(1.0 - float(_b[0]), 1e-6)
                prog = float(min(_ev / _nz, 1.0)) if _nz > 0.05 else _ev
            else:
                prog = float(out["progress_reward"])
            reward = self._rbm_weight * (prog - self._rbm_prev_prog)
            self._rbm_prev_prog = prog
            self._last_progress = prog
            self.episode_stats["vlm_robometer_progress_sum"] = prog
            if self._rbm_log_n < 12 or self._rbm_log_n % 500 == 0:
                print(f"[RBM-HOP] i={self._rbm_step_i} prog={prog:.4f} hop={reward:+.4f} "
                      f"trunc={time_truncated}", flush=True)
            self._rbm_log_n += 1
        if time_truncated and self._rbm_anchor:
            out = self.scorer(self.current_video, task=self.task_description,
                              icl_frames=self.icl_frames)
            sp = float(out["success_prob"])
            self._last_success_prob = sp
            anchor = sp * self.robometer_reward_scale
            if self.robometer_threshold > 0:
                anchor = 1.0 if anchor > self.robometer_threshold else 0.0
            reward += anchor
            self.episode_stats["vlm_robometer_success_prob_sum"] += sp
            if self._rbm_log_n < 24 or self._rbm_log_n % 500 == 1:
                print(f"[RBM-HOP] truncation anchor: succ={sp:.4f} -> +{anchor:.4f} "
                      f"(total={reward:+.4f})", flush=True)
        self.episode_stats["vlm_reward_counts"] += 1
        return rl_obs, float(reward), bool(time_truncated), success, image_obs

    def reset(self):
        rl_obs, image_obs = super().reset()
        self.current_video = [self.fetch_img(image_obs)]
        self.episode_stats = {
            "vlm_reward_TPR": 0,
            "vlm_reward_FPR": 0,
            "vlm_reward_TNR": 0,
            "vlm_reward_FNR": 0,
            "vlm_reward_counts": 0,
            "early_termination": 0,
            # Robometer-style critics expose progress + success heads; sum
            # them per-episode so the train loop can divide by reward_counts
            # to get a per-episode mean (matches the existing CM pattern).
            # Zero for non-Robometer VLMs.
            "vlm_robometer_progress_sum": 0.0,
            "vlm_robometer_success_prob_sum": 0.0,
        }
        self._last_progress = 0.0
        self._last_success_prob = 0.0
        self._consec_success = 0
        self._cur_ep_peak = 0.0
        self.vid_t = 1
        if self.vlm_name == "robodopamine_4b":
            self._rd_reset_episode()
        if self.vlm_name == "lrm_progress_8b":
            self._lrm_reset_episode()
        if getattr(self, "_rbm_interval", 0):
            self._rbm_prev_prog = 0.0
            self._rbm_step_i = 0
        return rl_obs, image_obs

    def step(self, action):
        rl_obs, reward, terminal, success, image_obs = super().step(action)
        current_img = self.fetch_img(image_obs)
        self.current_video.append(current_img)
        self.vid_t += 1
        if self.vlm_name == "robodopamine_4b":
            # Robo-Dopamine has its own multi-view, dense, non-autonomous path.
            return self._rd_step_reward(rl_obs, success, image_obs)
        if self.vlm_name == "lrm_progress_8b":
            return self._lrm_step_reward(rl_obs, success, image_obs)
        if getattr(self, "_rbm_interval", 0) and os.environ.get("RBM_MODE", "add") == "replace" and (
            "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft"
        ):
            # replace-mode: hops + optional anchor INSTEAD of the stock reward path
            # (no autonomous detection). Default RBM_MODE=add keeps the stock path,
            # including the autonomous protocol, and adds the hop term at return.
            return self._rbm_step_reward(rl_obs, success, image_obs)
        idx = past_frames_single_video(self.vid_t, self.past_len)

        if self.autonomous:
            # ---- FAIR autonomous protocol: the MODEL detects success, not GT.
            # Score the growing buffer every step (sets _last_success_prob).
            if self.single_frame_tile and (
                "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft"
            ):
                # Robometer-authors LIBERO mode: score ONLY the current frame,
                # tiled to max_frames. No temporal window -> per-state progress.
                tile_n = self.single_frame_tile if self.single_frame_tile > 1 \
                    else int(getattr(self.scorer, "max_frames", 16))
                vlm_reward = self.vlm_reward([current_img] * tile_n)
            elif "roboreward" in self.vlm_name or "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft":
                vlm_reward = self.vlm_reward(self.current_video)
            else:
                subsampled_video = [self.current_video[i] for i in idx]
                subsampled_video.append(current_img)
                vlm_reward = self.vlm_reward(subsampled_video)
            # The chosen detection head crossing the threshold for N consecutive
            # steps == success + terminate. Head selectable (success_prob vs
            # progress_reward); GT is NOT used either way.
            detect_value = (self._last_progress if self.detect_head == "progress"
                            else self._last_success_prob)
            self._cur_ep_peak = max(self._cur_ep_peak, float(detect_value))
            # DYNAMIC THRESHOLD (online lever; OFF unless ROBOMETER_DYNAMIC_THR=1).
            # No hand-calibrated threshold: the bar is the rolling ~q-th quantile
            # of recent ON-POLICY detection scores + margin (floored at
            # ROBOMETER_SUCCESS_THRESHOLD). When the policy adversarially inflates
            # scores, the quantile rises with it — the bar self-raises under
            # exactly the pressure that breaks a static threshold.
            # Maintain the rolling on-policy score buffer for any stream-based
            # online lever (DYNAMIC_THR quantile or OTSU bimodal valley).
            if self.dynamic_thr:
                self._dyn_scores.append(float(detect_value))
                if len(self._dyn_scores) > self.dyn_window:
                    self._dyn_scores = self._dyn_scores[-self.dyn_window:]
            if self.demo_anchor and self._demo_anchor_thr is not None:
                # DEMO-ANCHORED (demo-based): bar set once from demo success scores.
                eff_threshold = self._demo_anchor_thr
            elif self.otsu:
                # OTSU (demo-free): bar = Otsu valley over PER-EPISODE PEAK scores
                # (updated once per episode at ep_end, below). Failure-episodes peak
                # low, success-episodes peak high -> a real valley to fire at. Only
                # when the peaks are clearly BIMODAL (d >= otsu_d, i.e. a success
                # cluster has emerged); otherwise stay silent (bar=1.1) so we don't
                # fire before the policy ever succeeds. Demo-free, GT-free.
                if len(self._ep_peak_scores) >= self.otsu_min_eps:
                    _thr, _d, _mu_hi, _sd_hi = _otsu_threshold(self._ep_peak_scores, self.otsu_bins)
                    eff_threshold = max(self.success_threshold, _thr) if _d >= self.otsu_d else 1.1
                    # diagnostic: print once whenever the ep-peak buffer grows by 10
                    # (vid_t resets every episode & caps at 200, so a vid_t%500 gate
                    # never fires — use the buffer length as the throttle instead).
                    if len(self._ep_peak_scores) >= self._otsu_next_log:
                        self._otsu_next_log = len(self._ep_peak_scores) + 10
                        _a = np.asarray(self._ep_peak_scores)
                        print(f"[otsu] ep-peaks n={len(_a)} [min/med/max]="
                              f"{_a.min():.3f}/{np.median(_a):.3f}/{_a.max():.3f} "
                              f"valley={_thr:.3f} d={_d:.2f} mu_hi={_mu_hi:.3f} -> bar={eff_threshold:.3f}",
                              flush=True)
                else:
                    eff_threshold = 1.1  # warmup: still collecting per-episode peaks
            elif self.dynamic_thr:
                if len(self._dyn_scores) >= 50:
                    q = float(np.quantile(np.asarray(self._dyn_scores), self.dyn_q))
                    eff_threshold = max(self.success_threshold, q + self.dyn_margin)
                else:
                    eff_threshold = 1.1  # warmup: never fire on <50 observed scores
            else:
                eff_threshold = self.success_threshold
            # MIN-EPISODE-LENGTH GATE (online lever; OFF unless min_ep_steps>0):
            # real success cannot happen faster than the expert demo — ignore any
            # fire earlier than ~frac×(median demo length). Kills the 11–21-step
            # exploit fires categorically; zero-shot (demo lengths already known).
            gate_open = self.vid_t >= self.min_ep_steps
            if detect_value > eff_threshold and gate_open:
                self._consec_success += 1
            else:
                self._consec_success = 0
            model_success = self._consec_success >= self.success_consecutive
            # DUAL-HEAD AGREEMENT (demo-free FP filter; OFF unless thr>0): a real
            # success has BOTH heads high; a success-head false positive often has
            # mid progress (the leaky 0.3-0.7 range) -> veto it. Orthogonal signal,
            # free at runtime (progress head already computed each step).
            if model_success and self.progress_confirm_thr > 0.0 and \
                    self._last_progress < self.progress_confirm_thr:
                model_success = False
                self._consec_success = 0
            if model_success and self.dynamic_thr:
                print(f"[dyn-thr] fire at t={self.vid_t} score={detect_value:.3f} "
                      f"bar={eff_threshold:.3f} (q{self.dyn_q}+{self.dyn_margin})", flush=True)
            # CONFIRMATION VOTE (stability lever; OFF unless ROBOMETER_CONFIRM_K>0).
            # A candidate fire must be confirmed by a majority of K extra scores,
            # each on a jittered subsample of the buffer (~confirm_frac of frames
            # kept, last frame always included → the internal linspace-16 picks a
            # different frame set per vote). Real success is persistent across
            # subsamples; subsample-dependent false positives get vetoed. Cost:
            # K extra scorer calls ONLY at candidate fires.
            if model_success and self.confirm_k > 0 and (
                "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft"
            ):
                votes_yes = 1  # the original crossing counts as one ballot
                buf = self.current_video
                for _ in range(self.confirm_k):
                    n_keep = max(6, int(len(buf) * self.confirm_frac))
                    if n_keep >= len(buf):
                        keep = list(range(len(buf)))
                    else:
                        keep = sorted(random.sample(range(len(buf) - 1), n_keep - 1)) + [len(buf) - 1]
                    ballot_frames = [buf[i] for i in keep]
                    # PIXEL-AUGMENTATION ensemble (demo-free): perturb the kept frames
                    # (brightness/contrast jitter + small gaussian noise). An adversarial
                    # false positive that hacks specific pixels is BRITTLE to this and
                    # flips; a genuine success is robust. Adds *information* the
                    # frame-subsample vote can't (it re-queries the same pixels).
                    if self.confirm_augment > 0.0:
                        ballot_frames = [self._augment_frame(f) for f in ballot_frames]
                    out = self.scorer(ballot_frames, task=self.task_description,
                                      icl_frames=self.icl_frames)
                    v = (float(out["progress_reward"]) if self.detect_head == "progress"
                         else float(out["success_prob"]))
                    votes_yes += int(v > self.success_threshold)
                # unanimous (all ballots agree) vs strict majority
                need = (self.confirm_k + 1) if self.confirm_unanimous else (self.confirm_k + 1) // 2 + 1
                if votes_yes < need:
                    print(f"[confirm] VETO {votes_yes}/{self.confirm_k + 1} (need {need}) "
                          f"at t={self.vid_t}", flush=True)
                    model_success = False
                    self._consec_success = 0
                else:
                    print(f"[confirm] pass {votes_yes}/{self.confirm_k + 1} at t={self.vid_t}",
                          flush=True)
            time_truncated = bool(self.most_recent_info["truncated"])  # episode_length timeout
            ep_end = bool(model_success) or time_truncated
            # OTSU: record this episode's PEAK detection score (one value/episode) into
            # the rolling buffer the threshold is derived from. Done at ep_end so the
            # peak is complete; rolling window keeps it adaptive to on-policy drift.
            if self.otsu and ep_end:
                self._ep_peak_scores.append(float(self._cur_ep_peak))
                if len(self._ep_peak_scores) > self.otsu_win_eps:
                    self._ep_peak_scores = self._ep_peak_scores[-self.otsu_win_eps:]
            # reward_at_truncation: dense (rt=0) vs only at the terminating step (rt=1)
            out_reward = vlm_reward if (not self.reward_at_truncation or ep_end) else 0.0
            # OUTPUT SAMPLING (stability lever; OFF unless ROBOMETER_REWARD_SAMPLE=1):
            # emit Bernoulli(p) instead of the raw value, so a systematic false
            # positive stops paying out deterministically (Goodhart-resistance).
            if self.reward_sample and out_reward > 0.0:
                p = min(max(float(out_reward), 0.0), 1.0)
                out_reward = float(np.random.rand() < p)
            # diagnostics only (do NOT drive learning)
            self.episode_stats["vlm_reward_counts"] += 1
            if model_success and not terminal:
                self.episode_stats["early_termination"] += 1
            if "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft":
                self.episode_stats["vlm_robometer_progress_sum"] += self._last_progress
                self.episode_stats["vlm_robometer_success_prob_sum"] += self._last_success_prob
            # NOTE: returned `success` is GT, for logging only; the policy learns
            # from out_reward + the terminal flag, both model/timeout-driven (no GT).
            # Terminal on model-detection OR episode_length timeout — NOT on GT.
            # (Dropping the timeout here overflows the C++ replay buffer's
            # maxSeqLen_ when the model never fires; the train loop ends an
            # episode only on `terminal`.)
            return rl_obs, out_reward + self._rbm_hop_term(), bool(ep_end), success, image_obs

        if not self.end_on_success:
            truncated = self.most_recent_info["truncated"]
        else:
            truncated = self.most_recent_info["truncated"] or (reward == 1)

        if self.reward_at_truncation and not truncated:
            vlm_reward = 0.0
        else:
            if "roboreward" in self.vlm_name or "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft":
                # RoboReward and Robometer consume the full video. Robometer's
                # internal collator (linspace_subsample_frames) reduces to its
                # configured max_frames (16). Passing only past_len+1=5 frames
                # collapses the trained success signal — verified empirically
                # on a known-success training trajectory:
                #   16-frame input → success_prob ≈ 0.59 (correct)
                #    5-frame input → success_prob ≈ 0.009 (collapsed)
                vlm_reward = self.vlm_reward(self.current_video)
            else:
                # Demo2Reward and GVL consume a sparsely sampled set of frames.
                subsampled_video = [self.current_video[i] for i in idx]
                subsampled_video.append(current_img)
                vlm_reward = self.vlm_reward(subsampled_video)

        if not self.reward_at_truncation or (self.reward_at_truncation and truncated):
            reward_match = float(vlm_reward == reward)
            self.episode_stats["vlm_reward_counts"] += 1
            if reward:
                if reward_match:
                    self.episode_stats["vlm_reward_TPR"] += 1
                else:
                    self.episode_stats["vlm_reward_FNR"] += 1
            else:
                if reward_match:
                    self.episode_stats["vlm_reward_TNR"] += 1
                else:
                    self.episode_stats["vlm_reward_FPR"] += 1
            if "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft":
                self.episode_stats["vlm_robometer_progress_sum"] += self._last_progress
                self.episode_stats["vlm_robometer_success_prob_sum"] += self._last_success_prob

        if self.end_on_success and not self.most_recent_info["truncated"]:
            vlm_terminal = False
            if vlm_reward == 1:
                if not terminal:
                    self.episode_stats["early_termination"] += 1
                vlm_terminal = True
        else:
            vlm_terminal = terminal

        return rl_obs, vlm_reward + self._rbm_hop_term(), vlm_terminal, success, image_obs

    def vlm_reward(self, frames, debug=False):
        if "gvl" in self.vlm_name:
            demo_id = random.randint(0, 2)
            example_demo = self.all_demos[demo_id]
            init_frame = frames[0]
            rest_frames = frames[1:]
            return reward_from_GVL(
                model=self.vlm,
                processor=self.processor,
                prompt_vlm=self.prompt_vlm,
                task_description=self.task_description,
                example_frames=example_demo,
                example_percs=self.gvl_percs,
                test_init_frame=init_frame,
                test_frames=rest_frames,
            )

        if "robometer" in self.vlm_name or self.vlm_name == "qwen35_ft":
            out = self.scorer(frames, task=self.task_description, icl_frames=self.icl_frames)
            self._last_progress = float(out["progress_reward"])
            self._last_success_prob = float(out["success_prob"])
            mixed = self.robometer_beta * self._last_progress + (1.0 - self.robometer_beta) * self._last_success_prob
            mixed *= self.robometer_reward_scale
            if self.robometer_threshold > 0:
                reward = 1.0 if mixed > self.robometer_threshold else 0.0
            else:
                reward = mixed
            if debug:
                print(
                    f"Robometer: progress={self._last_progress:.4f} "
                    f"success={self._last_success_prob:.4f} "
                    f"beta={self.robometer_beta:.2f} scale={self.robometer_reward_scale:g} "
                    f"mixed={mixed:.4f} thr={self.robometer_threshold:.4f} → reward={reward:.3f}"
                )
            return reward

        is_roboreward = "roboreward" in self.vlm_name
        if is_roboreward:
            # RoboReward is a Qwen3-VL video scorer with a fixed token budget:
            # feeding the full rollout collapses spatial resolution to ~4x4 patches
            # (unreadable) AND floods the output with timestamp tokens. Subsample to
            # ROBOREWARD_NFRAMES (default 16, final frame always included) to keep a
            # readable resolution and a clean "ANSWER: <n>". Validated to reproduce
            # the paper's MetaWorld discrimination (BoxClose AUC~1.0, CoffeePush~0.68)
            # on their own eval clips; the full-frame feed gave a flat constant score.
            nf = int(os.environ.get("ROBOREWARD_NFRAMES", "16"))
            if len(frames) > nf:
                sel = np.linspace(0, len(frames) - 1, nf).round().astype(int)
                frames = [frames[i] for i in sel]
        critic_output = single_prompt_eval(
            prompt_vlm=self.prompt_vlm,
            model=self.vlm,
            processor=self.processor,
            system_prompt=self.system_prompt,
            prompt=self.prompt,
            frames=frames,
            debug=debug,
            use_video=is_roboreward,
            max_new_tokens=24 if is_roboreward else 5,
        )
        if is_roboreward:
            reward = roboreward_to_reward(critic_output)
        else:
            reward = response_to_reward(critic_output)
            # Generative binary success detectors (Demo2Reward, VLM-SD) expose their
            # 0/1 verdict as the detection signal so the AUTONOMOUS branch can
            # terminate the episode on detected success (threshold ~0.5), exactly
            # like the reward-head models. This is Demo2Reward's intended protocol:
            # a frozen VLM success detector that fires the moment it sees success.
            self._last_success_prob = float(reward)
        if debug:
            print("Reward =", reward)
        return reward
