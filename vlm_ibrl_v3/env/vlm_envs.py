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
from env.roboreward_utils import get_roboreward_8b, prompt_roboreward
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
        elif vlm in ("robometer_4b", "robometer_ft", "qwen35_ft"):
            # Robometer-family critics (progress + success heads, not
            # generative). All variants share one loader; only the
            # checkpoint path differs. FT variants read their path from an
            # env var so the SLURM job stays generic.
            _ckpt_map = {
                "robometer_4b": "robometer/Robometer-4B",
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
        self.vid_t = 1
        return rl_obs, image_obs

    def step(self, action):
        rl_obs, reward, terminal, success, image_obs = super().step(action)
        current_img = self.fetch_img(image_obs)
        self.current_video.append(current_img)
        self.vid_t += 1
        idx = past_frames_single_video(self.vid_t, self.past_len)

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

        return rl_obs, vlm_reward, vlm_terminal, success, image_obs

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
        critic_output = single_prompt_eval(
            prompt_vlm=self.prompt_vlm,
            model=self.vlm,
            processor=self.processor,
            system_prompt=self.system_prompt,
            prompt=self.prompt,
            frames=frames,
            debug=debug,
            use_video=is_roboreward,
            max_new_tokens=5,
        )
        if is_roboreward:
            reward = roboreward_to_reward(critic_output)
        else:
            reward = response_to_reward(critic_output)
        if debug:
            print("Reward =", reward)
        return reward
