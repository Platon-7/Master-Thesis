import os
import time
from collections import defaultdict, deque

import numpy as np
import robosuite
import torch
from PIL import Image
# Reverted to old controller API to match pinned robosuite@de64fa5 + shipped 1.4.1 data.
#from robosuite.controllers import load_composite_controller_config
from robosuite.controllers import load_controller_config

from common_utils import ibrl_utils as utils
from env.qwen_utils import get_qwen3, get_qwen3_8b, prompt_qwen
from env.roboreward_utils import get_roboreward_4b, get_roboreward_8b, prompt_roboreward
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import (
    ROBOMIMIC_DEMO2REWARD_REPLIES as demo2reward_replies,
    ROBOMIMIC_TASK_DESCRIPTIONS as task_description,
    roboreward_prompt,
    vlmcritic_prompt,
)


# Valid VLM names for VLMRobosuite. (GVL is not used in the RoboMimic figures of the paper.)
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
    "robometer_ft",
    "qwen35_ft",
)


# ---------------------------------------------------------------------------
# Helpers (duplicated with env/vlm_envs.py for now — see flags for step 3)
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
            "sample_fps": 60,
            "video_metadata": {"duration": len(frames) / 60.0},
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
    if len(text.strip()) != 1:
        return 0.0
    return float(text.strip()[0])


def roboreward_to_reward(text: str) -> float:
    score_str = extract_answer_after_substring(text, substring="ANSWER:")
    if score_str not in ["1", "2", "3", "4", "5"]:
        # malformed / truncated generation → neutral reward rather than crash the RL run
        print(f"[roboreward] unparseable answer {score_str!r}; reward=0")
        return 0.0
    return (int(score_str) - 1) / 4.0  # normalize to [0, 1]


# ---------------------------------------------------------------------------
# Camera / state metadata (kept here so PixelRobosuite/VLMRobosuite share)
# ---------------------------------------------------------------------------

# all avail views:
# 'frontview', 'birdview', --> too far for this task
# 'agentview', 'robot0_robotview', --> same
# 'sideview', 'robot0_eye_in_hand'
GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_KEYS = {
    "Lift": DEFAULT_STATE_KEYS,
    "PickPlaceCan": DEFAULT_STATE_KEYS,
    "NutAssemblySquare": DEFAULT_STATE_KEYS,
    "TwoArmTransport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "ToolHang": [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ],
}
STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
}
PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PROP_DIM = 9


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img_tensor)}")
    img = img_tensor.detach().cpu()
    shape = img.shape
    if img.ndim == 3:
        if shape[0] == 3:
            img = img.permute(1, 2, 0)
        elif shape[2] == 3:
            pass
        else:
            raise ValueError(f"Ambiguous image shape {shape}")
    else:
        raise ValueError(f"Unsupported tensor shape {shape}")

    img = img.numpy()
    if img.dtype != np.uint8:
        print("Img float with", np.min(img), np.max(img))
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# VLM-critic-wrapped RoboMimic environment
# ---------------------------------------------------------------------------

class VLMRobosuite:
    """RoboMimic env wrapped with a frozen VLM reward model.

    Accepted ``vlm`` values:
        - ``vlm_sd_qwen3_8b`` / ``vlm_sd_qwen3_32b``           : VLM-SD zero-shot baseline (generic task instruction)
        - ``demo2reward_qwen3_8b`` / ``demo2reward_qwen3_32b`` : Demo2Reward critic with optimized prompts
        - ``roboreward_8b``                                    : RoboReward baseline
    """

    def __init__(
        self,
        env_name,
        robots,
        episode_length,
        vlm,
        *,
        reward_shaping=False,
        image_size=224,
        rl_image_size=96,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_cameras=["agentview"],
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        cond_action=0,
        flip_image=True,  # only false if using with eval_with_init_state
        ctrl_delta=True,
        record_sim_state: bool = False,
        past_len=4,
        reward_at_truncation=False,
        vlm_camera="agentview",
        robometer_beta: float = 0.0,
        robometer_threshold: float = 0.0,
        robometer_reward_scale: float = 1.0,
    ):
        assert vlm in VALID_VLMS, f"VLM {vlm} not recognized. Valid: {VALID_VLMS}"
        assert past_len > 0, "past_len must be > 0"
        assert isinstance(camera_names, list)
        # RoboDopamine renders multi-view frames on demand from the sim, so its cameras must
        # exist in the model. Inject them into camera_names before the env is built.
        if vlm == "robodopamine_4b":
            # PickPlaceCan exposes agentview + robot0_eye_in_hand; there is no 2nd wrist,
            # so reuse the wrist as the 3rd view (handoff-sanctioned fallback). camera_names
            # dedupes; _rd_cams keeps 3 entries for the scorer's 3-view input.
            self._rd_cams = os.environ.get(
                "ROBODOPAMINE_CAMS", "agentview,robot0_eye_in_hand,robot0_eye_in_hand").split(",")
            camera_names = list(dict.fromkeys(list(camera_names) + self._rd_cams))
        self.camera_names = camera_names
        self.ctrl_config = load_controller_config(default_controller="OSC_POSE")
        self.ctrl_config["control_delta"] = ctrl_delta
        self.record_sim_state = record_sim_state
        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=self.ctrl_config,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=reward_shaping,
            camera_names=self.camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            horizon=episode_length,
        )
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.flip_image = flip_image

        self.resize_transform = None
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)

        self.action_dim: int = len(self.env.action_spec[0])
        self._observation_shape: tuple[int, ...] = (3 * obs_stack, rl_image_size, rl_image_size)
        self._state_shape: tuple[int] = (STATE_SHAPE[env_name][0] * state_stack,)
        self.prop_shape: tuple[int] = (PROP_DIM * prop_stack,)
        self.device = device

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.obs_stack = obs_stack
        self.state_stack = state_stack
        self.prop_stack = prop_stack
        self.cond_action = cond_action
        self.past_obses = defaultdict(list)
        self.past_actions = deque(maxlen=self.cond_action)

        task = task_description[env_name]
        self.task_description = task

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
        elif vlm in ("roboreward_8b", "roboreward_4b"):
            self.vlm, self.processor = get_roboreward_4b() if vlm == "roboreward_4b" else get_roboreward_8b()
            self.prompt_vlm = prompt_roboreward
            self.system_prompt = roboreward_prompt
            self.prompt = f"{roboreward_prompt}\n\nTask: {task}"
        elif vlm == "robodopamine_4b":
            # Robo-Dopamine GRM: generative, MULTI-VIEW, hop-based progress. Per-episode
            # scorer (bound to goal + reference-start frame) built in reset(); dense
            # potential-based DELTA reward scored every ROBODOPAMINE_STRIDE steps.
            from env.robodopamine_utils import get_robodopamine
            _rd_path = os.environ.get("ROBODOPAMINE_PATH",
                                      "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview")
            self.vlm, self.processor = get_robodopamine(_rd_path)
            self._rd_stride = int(os.environ.get("ROBODOPAMINE_STRIDE", "16"))
            self._rd_eval_mode = os.environ.get("ROBODOPAMINE_EVAL_MODE", "forward")
            self._rd_goal_path = os.environ.get("ROBODOPAMINE_GOAL", "")
            self._rd_res = int(os.environ.get("ROBODOPAMINE_RES", "224"))
            self._rd_task = task
            self.prompt_vlm = self.system_prompt = self.prompt = None
        elif vlm == "lrm_progress_8b":
            # LRM (Large Reward Models, TRI/USC-PSI) single-frame progress, Qwen3-VL-8B ->
            # [0,1]. Needs the episode's frame-0 INITIAL anchor (zero-shot). Reward form is
            # DELTA for our off-policy IBRL (absolute-held is a survival-bonus trap, handoff §3).
            from env.lrm_utils import get_lrm_progress
            self.vlm, self.processor = get_lrm_progress()
            self._lrm_interval = int(os.environ.get("LRM_CALL_INTERVAL", "10"))
            self._lrm_res = int(os.environ.get("LRM_RES", "256"))
            self._lrm_reward_mode = os.environ.get("LRM_REWARD_MODE", "delta")
            self._lrm_cam = os.environ.get("LRM_CAM", "agentview")
            self._lrm_use_initial = os.environ.get("LRM_INCLUDE_INITIAL", "1") == "1"
            self._lrm_task = task
            self.prompt_vlm = self.system_prompt = self.prompt = None
        elif vlm in ("robometer_4b", "robometer_ft", "qwen35_ft"):
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
        else:
            raise AssertionError(f"VLM {vlm} not recognized")  # unreachable

        self.vlm.eval()
        self.current_video = []
        self.vid_t = 0
        self.past_len = past_len
        self.vlm_name = vlm
        self.vlm_camera = vlm_camera

        self.reward_at_truncation = reward_at_truncation
        # Robometer reward composition: reward = beta * progress + (1 - beta)
        # * success_prob. 0.0 = pure success_prob. Ignored for non-Robometer.
        self.robometer_beta = float(robometer_beta)
        # Optional binarization (see env/vlm_envs.py for rationale).
        self.robometer_threshold = float(robometer_threshold)
        # Optional reward scale applied before thresholding.
        self.robometer_reward_scale = float(robometer_reward_scale)
        self._last_progress = 0.0
        self._last_success_prob = 0.0

        # ---- FAIR autonomous-RL mode (ported from env/vlm_envs.py, metaworld) ----
        # The MODEL detects success and terminates the episode; GT is NEVER used for
        # termination or reward (returned only as a logged diagnostic — the separate
        # GT eval_env measures honest performance). PROVEN CORE ONLY: detection head +
        # threshold + consecutive-step debounce. The online-calibration levers
        # (OTSU / demo-anchor / dynamic-quantile) are intentionally NOT ported yet —
        # they're still being verified; they will plug in at _effective_threshold().
        self.autonomous = os.environ.get("AUTONOMOUS_SUCCESS", "0") == "1"
        self.success_threshold = float(os.environ.get("ROBOMETER_SUCCESS_THRESHOLD", "0.6"))
        self.success_consecutive = int(os.environ.get("ROBOMETER_SUCCESS_CONSECUTIVE", "1"))
        # which head drives DETECTION (independent of the reward beta-mix):
        # "success" = success_prob (default), "progress" = progress_reward.
        self.detect_head = os.environ.get("ROBOMETER_DETECT_HEAD", "success")
        self._consec_success = 0
        # MIN-EPISODE-LENGTH GATE (ported from vlm_envs.py): a detection earlier than
        # min_ep_steps is ignored — real success cannot happen faster than ~a demo, so
        # this categorically kills the early-fire reward-hack (e.g. the ~31-step exploit
        # on PickPlaceCan). 0 = disabled (default). Zero-shot: needs no GT/demos at runtime.
        self.min_ep_steps = int(os.environ.get("ROBOMETER_MIN_EP_STEPS", "0"))
        if self.autonomous:
            print(f"[autonomous] head={self.detect_head} thr={self.success_threshold} "
                  f"consec={self.success_consecutive} min_ep_steps={self.min_ep_steps} "
                  f"(offline-fixed threshold seam)", flush=True)

        # ICL context (ported from vlm_envs.py): when ROBOMETER_ICL_DEMO_PATH is set
        # (a dir of `{demo_idx}_NNN.png` frames), load N uniform frames of the chosen
        # demo and pass them as the in-context demonstration to every scorer call.
        self.icl_frames = None
        icl_path = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
        if icl_path and "robometer" in self.vlm_name:
            from pathlib import Path as _P
            from PIL import Image as _PIL
            icl_idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
            icl_n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
            avail = sorted(p for p in _P(icl_path).iterdir()
                           if p.name.startswith(f"{icl_idx}_") and p.suffix == ".png")
            if not avail:
                raise FileNotFoundError(f"ROBOMETER_ICL_DEMO_PATH={icl_path} has no frames for demo {icl_idx}")
            picks = np.linspace(0, len(avail) - 1, icl_n).round().astype(int)
            self.icl_frames = [np.asarray(_PIL.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]
            print(f"[ICL] loaded {icl_n} frames from demo {icl_idx} of {icl_path} (of {len(avail)})", flush=True)

        assert len(self.rl_cameras) == 1

    def _effective_threshold(self, detect_value):
        """Detection-threshold seam. SAFE DEFAULT = offline-fixed ROBOMETER_SUCCESS_THRESHOLD.
        The online-calibration rules (OTSU / demo-anchor / dynamic-quantile) plug in HERE
        once verified on metaworld, so robomimic gets them as a one-method change with no
        other edits to step(). See [[online-calibration-direction]]."""
        return self.success_threshold

    @property
    def observation_shape(self):
        if self.use_state:
            return self._state_shape
        return self._observation_shape

    def _extract_images(self, obs):
        high_res_images = {}
        rl_obs = {}

        if self.use_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.state_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])

            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        if self.record_sim_state:
            sim_state = self.env.sim.get_state().flatten()
            rl_obs["sim_state"] = torch.from_numpy(sim_state)
            for key in DEFAULT_STATE_KEYS:
                env_key = "object-state" if key == "object" else key
                rl_obs[key] = torch.from_numpy(obs[env_key])

        return rl_obs, high_res_images

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
        self.current_video.clear()

        for _ in range(self.cond_action):
            self.past_actions.append(torch.zeros(self.action_dim))

        obs = self.env.reset()
        rl_obs, high_res_images = self._extract_images(obs)
        current_img = tensor_to_pil(high_res_images[self.vlm_camera])
        self.current_video.append(current_img)
        self.vid_t = 0

        if self.cond_action > 0:
            past_action = torch.from_numpy(np.stack(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        if self.vlm_name == "robodopamine_4b":
            self._rd_reset_episode(high_res_images)
        elif self.vlm_name == "lrm_progress_8b":
            self._lrm_reset_episode(high_res_images)

        return rl_obs, high_res_images

    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """All inputs and outputs are tensors."""
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs = {}
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        truncation = False
        current_img = None
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _ = self.env.step(actions[i])
            curr_rl_obs, curr_high_res_images = self._extract_images(obs)

            current_img = tensor_to_pil(curr_high_res_images[self.vlm_camera])
            self.current_video.append(current_img)

            self.vid_t += 1

            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                # In autonomous mode, GT success must NOT terminate the episode
                # (the model decides). `terminal` then reflects only the env horizon
                # timeout. `success` is still tracked for logging.
                if self.end_on_success and not self.autonomous:
                    terminal = True

            if not self.end_on_success and terminal:
                truncation = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal

        # Dense multi-step scorers (non-autonomous): score every N steps, deliver a
        # potential-based delta. Terminal only on the horizon timeout (no GT leak).
        if self.vlm_name == "robodopamine_4b":
            return self._rd_step_reward(rl_obs, success, high_res_images, truncation)
        if self.vlm_name == "lrm_progress_8b":
            return self._lrm_step_reward(rl_obs, success, high_res_images, truncation)

        if self.autonomous:
            # Score the growing video EVERY step → sets _last_success_prob/_last_progress.
            vlm_reward = self.vlm_reward(self.current_video)
            detect_value = (self._last_progress if self.detect_head == "progress"
                            else self._last_success_prob)
            eff_threshold = self._effective_threshold(detect_value)
            # min-episode-length gate: a fire before min_ep_steps can't be a real success.
            gate_open = self.time_step >= self.min_ep_steps
            if detect_value > eff_threshold and gate_open:
                self._consec_success += 1
            else:
                self._consec_success = 0
            model_success = self._consec_success >= self.success_consecutive
            time_truncated = bool(terminal)  # env horizon timeout (GT success no longer sets terminal)
            ep_end = bool(model_success) or time_truncated
            # reward_at_truncation: dense (rt=0) vs only at the terminating step (rt=1)
            out_reward = vlm_reward if (not self.reward_at_truncation or ep_end) else 0.0
            self.terminal = ep_end
            # `success` (GT) returned for LOGGING ONLY; the policy learns from
            # out_reward + ep_end, both model/timeout-driven (never GT).
            return rl_obs, out_reward, bool(ep_end), success, high_res_images

        # VLM reward
        if self.reward_at_truncation and not truncation:
            vlm_reward = 0.0
        elif "robometer" in self.vlm_name:
            # Robometer consumes the full video (its own internal subsampling handles it).
            vlm_reward = self.vlm_reward(self.current_video)
        elif "roboreward" in self.vlm_name:
            # RoboReward: subsample to 16 frames (final frame always included) — feeding the
            # full rollout collapses spatial res to 4x4 and floods timestamp tokens (handoff §2).
            nfr = int(os.environ.get("ROBOREWARD_NFRAMES", "16"))
            vid = self.current_video
            if len(vid) > nfr:
                idx = np.linspace(0, len(vid) - 1, nfr).round().astype(int)
                vid = [vid[i] for i in idx]
            vlm_reward = self.vlm_reward(vid)
        else:
            # Demo2Reward consumes a sparsely sampled set of frames.
            idx = past_frames_single_video(self.vid_t, self.past_len)
            subsampled_video = [self.current_video[i] for i in idx]
            subsampled_video.append(current_img)
            vlm_reward = self.vlm_reward(subsampled_video)

        return rl_obs, vlm_reward, terminal, success, high_res_images

    def vlm_reward(self, frames, debug=False):
        if "robometer" in self.vlm_name:
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
            max_new_tokens=(24 if is_roboreward else 5),
        )
        if is_roboreward:
            reward = roboreward_to_reward(critic_output)
        else:
            reward = response_to_reward(critic_output)
        if debug:
            print("Reward =", reward)
        return reward

    # ---- RoboDopamine (multi-view, dense, potential-based delta) ----------------
    def _rd_views_from(self, high_res_images):
        """The RoboDopamine cameras are in camera_names, so they are already rendered into
        high_res_images each step (CHW uint8 tensors) — pull them as PIL, no extra render."""
        return [tensor_to_pil(high_res_images[c]) for c in self._rd_cams]

    def _rd_reset_episode(self, high_res_images):
        from env.robodopamine_utils import RoboDopamineScorer
        start_views = self._rd_views_from(high_res_images)
        if self._rd_goal_path and os.path.exists(self._rd_goal_path):
            goal_img = Image.open(self._rd_goal_path).convert("RGB")
        else:
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

    def _rd_step_reward(self, rl_obs, success, high_res_images, truncation):
        from env.robodopamine_utils import accumulate_progress
        self._rd_step_i += 1
        reward = 0.0
        if self._rd_step_i % self._rd_stride == 0 or truncation:
            after_views = self._rd_views_from(high_res_images)
            before = (self._rd_start_views if self._rd_eval_mode == "forward"
                      else self._rd_prev_views)
            raw = self._rd_scorer.score(before, after_views)
            if raw is not None:
                prog = accumulate_progress(self._rd_eval_mode, raw, self._rd_prev_prog)
                reward = prog - self._rd_prev_prog          # potential-based hop
                self._rd_prev_prog = prog
                self._last_progress = prog
            self._rd_prev_views = after_views
        return rl_obs, float(reward), bool(truncation), success, high_res_images

    # ---- LRM (single-frame progress, potential-based delta) --------------------
    def _lrm_frame_from(self, high_res_images):
        img = tensor_to_pil(high_res_images[self._lrm_cam])
        if img.size != (self._lrm_res, self._lrm_res):
            img = img.resize((self._lrm_res, self._lrm_res))  # >= LRM's 256^2 min_pixels
        return img

    def _lrm_reset_episode(self, high_res_images):
        from env.lrm_utils import LRMProgressScorer
        init_img = self._lrm_frame_from(high_res_images) if self._lrm_use_initial else None
        self._lrm_scorer = LRMProgressScorer(self.vlm, self.processor,
                                             task=self._lrm_task, initial_img=init_img)
        self._lrm_held = 0.0
        self._lrm_prev_prog = 0.0
        self._lrm_step_i = 0

    def _lrm_step_reward(self, rl_obs, success, high_res_images, truncation):
        self._lrm_step_i += 1
        reward = self._lrm_held  # "hold" carries the last value between calls
        if self._lrm_step_i % self._lrm_interval == 0 or truncation:
            prog = max(0.0, min(1.0, float(self._lrm_scorer.score(self._lrm_frame_from(high_res_images)))))
            if self._lrm_reward_mode == "delta":
                reward = prog - self._lrm_prev_prog
                self._lrm_prev_prog = prog
            else:
                self._lrm_held = prog
                reward = prog
            self._last_progress = prog
        elif self._lrm_reward_mode == "delta":
            reward = 0.0  # delta pays out only at scored steps
        return rl_obs, float(reward), bool(truncation), success, high_res_images
