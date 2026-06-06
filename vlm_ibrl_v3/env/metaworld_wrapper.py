import re
import torch
import numpy as np
import collections

try:
    import gym  # old gym (0.26): the IBRL wrapper stack below subclasses gym.Env / gym.Wrapper
    import metaworld  # Farama MetaWorld v3 (gymnasium-based); see MetaWorld/setup_env.sh pin 58e32b4d
    import metaworld.policies as mw_policies
except Exception as e:
    print("warning: failed to import metaworld (v3)")
    print("========================================", e)
    print("========================================")


GOOD_CAMERAS = {
    "Assembly": ["corner2"],
    "Basketball": ["corner2"],
    "CoffeePush": ["corner2"],
    "BoxClose": ["corner2"],
    "StickPull": ["corner2"],
    "StickPull": ["corner2"],
    "PegInsertSide": ["corner2"],
    "Soccer": ["corner2"],
}
DEFAULT_CAMERA = "corner2"


# All V2 environments have 39 dimensional states. Some of the
# dimensions are unused (always 0), but we keep them all.
STATE_IDXS = {
    "Assembly": list(range(39)),
    "Basketball": list(range(39)),
    "CoffeePush": list(range(39)),
    "BoxClose": list(range(39)),
    "HandInsert": list(range(39)),
    "StickPull": list(range(39)),
    "PegInsertSide": list(range(39)),
    "Soccer": list(range(39)),
}
STATE_SHAPE = {env_name: (len(STATE_IDXS[env_name]),) for env_name in STATE_IDXS.keys()}

# We can find out what state dimensions correspond to by inspecting
# the observation_space definition in the base SawyerXYZEnv,
# (metaworld/envs/mujoco/sawyer_xyz_env.py)
# For more per-environment information, can look at the oracle policy,
# (e.g. metaworld/policies/sawyer_assembly_v2_policy.py).
# For all V2 environments, the first four dimensions are x, y, z, gripper;
# though for some environments, the gripper is not necessary.
PROP_IDXS = {
    "Assembly": list(range(4)),
    "Basketball": list(range(4)),
    "CoffeePush": list(range(4)),
    "BoxClose": list(range(4)),
    "StickPull": list(range(4)),
    "HandInsert": list(range(4)),
    "PegInsertSide": list(range(4)),
    "Soccer": list(range(4)),
}
PROP_SHAPE = {env_name: (len(PROP_IDXS[env_name]),) for env_name in STATE_IDXS.keys()}


# MetaWorld VERSION NOTE: this wrapper targets Farama MetaWorld **v3**
# (gymnasium + the modern `mujoco` bindings) — the rendering domain the
# Robometer reward model was trained/evaluated on (curated ids
# `metaworld_*_v3_*`). It is a port of the original v2 wrapper (rlworkgroup
# metaworld 0.1.0 / mujoco_py / mujoco210); the v2->v3 swap fixes the
# rendering-domain mismatch documented in the IBRL investigation. Only this
# class talks to gymnasium/v3 — every wrapper further down stays on the
# old-gym 4-tuple interface, so MetaWorldEnv translates the gymnasium 5-tuple
# (obs, reward, terminated, truncated, info) back to (obs, reward, done, info)
# at this single boundary.

_V3_MT1_SEED = 42  # only fixes which train_tasks MT1 generates; per-reset goal
                   # randomization comes from `_freeze_rand_vec = False` below.


class MetaWorldEnv(gym.Env):
    """
    Fully-observable state-only (noimage) MetaWorld **v3** environment.
    `camera_name`, `width`, and `height` only affect the output of `render`.

    Rendering replicates the curated-data pipeline exactly
    (MetaWorld/generate_failures.py::render_all_cameras): point the gymnasium
    MujocoRenderer's `camera_id` at the requested camera, render the env's
    native rgb buffer, flip it vertically, then resize to (width, height).
    Matching this keeps on-policy frames in the reward model's training domain.
    """

    def __init__(
        self,
        env_name,
        camera_name,
        width,
        height,
    ):
        self.env_name = env_name

        # Convert, e.g., CoffeePush -> coffee-push-v3
        task_id = re.sub(r"([a-z])([A-Z])", r"\1-\2", self.env_name).lower()
        task_id = f"{task_id}-v3"
        self.task_id = task_id

        # Build via the MT1 benchmark API — the same entrypoint the v3 dataset
        # generator used (MetaWorld/generate_dataset.py). MT1 yields the env
        # class + a list of tasks (each task = a frozen object/goal rand_vec).
        self._mt1 = metaworld.MT1(task_id, seed=_V3_MT1_SEED)
        env_cls = self._mt1.train_classes[task_id]
        self._tasks = list(self._mt1.train_tasks)

        self.env = env_cls(render_mode="rgb_array", camera_name=camera_name)
        # Seed an initial task, then unfreeze so every `reset` re-randomizes the
        # object/goal placement (mirrors the v2 wrapper's `_freeze_rand_vec =
        # False`; without this, set_task pins a single layout forever).
        self.env.set_task(self._tasks[0])
        self.env._freeze_rand_vec = False

        # DUAL-RENDER support. The v2-trained BC policy expects the v2 wrapper's
        # zoomed-in corner2 (Seo/Hansen cam_pos) and is out-of-domain under v3's
        # *default* corner2 (empirically: BC success 0.40 zoomed vs 0.00 default).
        # The reward model is the opposite — it was trained on the DEFAULT corner2
        # (render_match_v3 pixel-match). So:
        #   - V3_CORNER2_ZOOM=1 zooms the real `corner2` camera -> the POLICY's
        #     rl_camera="corner2" obs (and the corner2_image BC demos) are in-domain.
        #   - the pseudo-camera "corner2_default" renders corner2's DEFAULT view ->
        #     the REWARD model's reward_camera is in-domain.
        # Point ROBOMETER_REWARD_CAMERA=corner2_default so policy and reward are
        # BOTH in-domain in the same v3 episode (no BC retrain needed).
        #
        # IMPLEMENTATION NOTE: gymnasium 0.29.1's cached offscreen viewer does NOT
        # pick up cam_pos changes made *after* the first render (a runtime swap
        # gave byte-identical frames), but it DOES honor changes made at INIT
        # (that is why the BC zoom works). So we configure a spare world-fixed
        # camera ("corner", unused by the IBRL pipeline) to hold corner2's DEFAULT
        # extrinsics/intrinsics at init, and render *that* for "corner2_default".
        import os as _os
        import mujoco as _mj
        _CAM = _mj.mjtObj.mjOBJ_CAMERA
        m = self.env.model
        self._corner2_id = _mj.mj_name2id(m, _CAM, "corner2")
        self._corner2_zoom = _os.environ.get("V3_CORNER2_ZOOM") == "1"
        # Spare camera that mirrors corner2's DEFAULT view (set BEFORE zooming corner2).
        self._reward_cam_name = "corner"  # world-fixed, otherwise unused
        _sid = _mj.mj_name2id(m, _CAM, self._reward_cam_name)
        m.cam_pos[_sid] = np.array(m.cam_pos[self._corner2_id]).copy()
        m.cam_quat[_sid] = np.array(m.cam_quat[self._corner2_id]).copy()
        m.cam_fovy[_sid] = float(m.cam_fovy[self._corner2_id])
        m.cam_mode[_sid] = int(m.cam_mode[self._corner2_id])
        m.cam_bodyid[_sid] = int(m.cam_bodyid[self._corner2_id])
        if self._corner2_zoom:
            m.cam_pos[self._corner2_id] = np.array([0.75, 0.075, 0.7])

        # Scripted (oracle) policy, e.g. SawyerCoffeePushV3Policy.
        self.heuristic_policy = getattr(mw_policies, f"Sawyer{self.env_name}V3Policy")()

        self.camera_name = camera_name
        self.width = width
        self.height = height

        # Latest raw 39-d observation (fed to the scripted policy each call).
        self._last_obs = None
        # camera-name -> mujoco camera id, looked up lazily.
        self._cam_id_cache = {}

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # Match the v2 wrapper: take one zero-action step so the returned obs is
        # the settled post-reset state. gymnasium returns a 5-tuple.
        obs, _, _, _, _ = self.env.step(np.zeros_like(self.env.action_space.sample()))
        self._last_obs = obs
        obs = np.take(obs, STATE_IDXS[self.env_name])
        return dict(state=obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        done = bool(terminated or truncated)
        obs = dict(state=obs)
        return obs, reward, done, info

    def get_heuristic_action(self, clip_action=True):
        action = self.heuristic_policy.get_action(self._last_obs)
        if clip_action:
            action = action.clip(-1, 1)
        return action

    def render(self, mode="rgb_array", camera_name=None, width=None, height=None):
        assert mode == "rgb_array"
        # Use the defaults we initialized the environment with
        # unless different values are specifically passed to `render`
        camera_name = camera_name or self.camera_name
        width = width or self.width
        height = height or self.height

        # Render the named camera via gymnasium MujocoRenderer's *argument* API
        # (renderer.render(..., camera_name=cam)) — the same path env.render()
        # uses internally. NOTE: the v3 dataset generator
        # (MetaWorld/generate_failures.py::render_all_cameras) tried to swap
        # cameras by setting a `renderer.camera_id` ATTRIBUTE, which this
        # gymnasium (0.29.1) ignores — its MujocoRenderer has no such attribute,
        # so that code always fell through to its `except: env.render()` branch.
        # The net effect: every curated frame is the env's construction camera
        # (corner2) rendered via env.render() and vertically flipped. We
        # reproduce that exactly here (camera_name + `[::-1]`) so on-policy
        # frames sit in the reward model's training domain. corner2 is the
        # in-domain camera; other names still render correctly but were never
        # seen by the reward model.
        #
        # Pseudo-camera "corner2_default" -> the spare camera configured at init
        # to hold corner2's DEFAULT (un-zoomed) view (the reward model's in-domain
        # view in the dual-render setup). It is an init-time config because
        # runtime cam_pos swaps do NOT propagate through gymnasium's cached
        # offscreen viewer (verified: a runtime swap produced byte-identical
        # frames).
        real_name = self._reward_cam_name if camera_name == "corner2_default" else camera_name
        img = self.env.mujoco_renderer.render("rgb_array", camera_name=real_name)
        img = img[::-1].copy()

        if img.shape[0] != height or img.shape[1] != width:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))
        return img



class ProprioObsWrapper(gym.Wrapper):
    """
    Takes a MetaWorld environment and adds an observation key
    for the proprioceptive state
    """

    def __init__(self, env, idx_list):
        super().__init__(env)
        self.idx_list = idx_list

    def _modify_observation(self, obs):
        obs["prop"] = np.take(obs["state"], self.idx_list)

    def reset(self):
        obs = self.env.reset()
        self._modify_observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._modify_observation(obs)
        return obs, reward, done, info


class ImageObsWrapper(gym.Wrapper):
    """
    Takes a MetaWorld environment and adds an observation key
    for image from one or more cameras
    """

    def __init__(self, env, camera_names):
        super().__init__(env)
        self.camera_names = camera_names

    def _modify_observation(self, obs):
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            img = self.env.render(camera_name=camera_name)
            obs[image_key] = img.transpose(2, 0, 1)

    def reset(self):
        obs = self.env.reset()
        self._modify_observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._modify_observation(obs)
        return obs, reward, done, info

class DualResImageObsWrapper(gym.Wrapper):
    """
    Takes a MetaWorld environment and adds an observation key
    for image from one or more cameras
    """

    def __init__(self, env, camera_names, higres_height=224, highres_width=224):
        super().__init__(env)
        self.camera_names = camera_names
        self.highres_height = higres_height
        self.highres_width = highres_width

    def _modify_observation(self, obs):
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            img = self.env.render(camera_name=camera_name)
            obs[image_key] = img.transpose(2, 0, 1)
            highres_img = self.env.render(camera_name=camera_name, height=self.highres_height, width=self.highres_width)
            obs[f"highres_{image_key}"] = highres_img.transpose(2, 0, 1)

    def reset(self):
        obs = self.env.reset()
        self._modify_observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._modify_observation(obs)
        return obs, reward, done, info


class ActionRepeatWrapper(gym.Wrapper):
    """
    Executes multiple inner `step` calls for each `step` call
    """

    def __init__(self, env, num_repeats):
        super().__init__(env)
        self.env = env
        self.num_repeats = num_repeats

    def step(self, action):
        obs = None
        reward = 0.0
        done = False
        info = None
        discount = 1.0
        for _ in range(self.num_repeats):
            _obs, _reward, _done, _info = self.env.step(action)
            obs = _obs
            reward += _reward * discount
            done = done or _done
            info = _info
            if done:
                break

        return obs, reward, done, info


class SparseRewardWrapper(gym.Wrapper):
    """
    Overwrite the default environment reward with binary success flag
    given in `info`. Does not overwrite the value of `done`.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # NOTE: This is different than MoDem, which uses a -1 / 0 reward
        # instead of 0, 1
        obs, reward, done, info = self.env.step(action)
        info["original_reward"] = reward
        reward = float(info["success"])
        return obs, reward, done, info

class TooEarlySparseRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # NOTE: This is different than MoDem, which uses a -1 / 0 reward
        # instead of 0, 1
        obs, reward, done, info = self.env.step(action)
        info["original_reward"] = reward
        reward = float(reward > 1)
        return obs, reward, done, info


class StackObsWrapper(gym.Wrapper):
    """
    Stacks observations from a history of the specified length.
    Keeps track of the history in `self.past_obses` and `self.past_frames`
    and returns stacked versions in `step`

    For non-image keys, concatenates along the first/only dimension
    For image keys, concatenates along the first/channel dimension
    """

    def __init__(self, env, obs_stack=1, frame_stack=1):
        super().__init__(env)
        self.obs_stack = obs_stack
        self.frame_stack = frame_stack
        self.past_obses = collections.defaultdict(lambda: collections.deque(maxlen=self.obs_stack))
        self.past_frames = collections.defaultdict(
            lambda: collections.deque(maxlen=self.frame_stack)
        )

    def _get_stacked_observation(self):
        # Concatenate along the first (only) dimension
        obses = {k: np.concatenate(v, axis=0) for k, v in self.past_obses.items()}
        # Concatenate along the first (channel) dimension
        frames = {k: np.concatenate(v, axis=0) for k, v in self.past_frames.items()}
        obses.update(frames)
        return obses

    def reset(self):
        obs = super().reset()
        self.past_obses.clear()
        self.past_frames.clear()

        # Fill up history with multiple copies of the first observation
        # NOTE: This consistent with what is done in the MoDem implementation
        # but not with L153 of robosuite_wrapper.py
        for key in obs:
            if "image" in key:
                for _ in range(self.frame_stack):
                    self.past_frames[key].append(obs[key].copy())
            else:
                for _ in range(self.obs_stack):
                    self.past_obses[key].append(obs[key].copy())

        return self._get_stacked_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for key in obs:
            if "image" in key:
                self.past_frames[key].append(obs[key])
            else:
                self.past_obses[key].append(obs[key])

        return self._get_stacked_observation(), reward, done, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        assert self._max_episode_steps > 0
        self._elapsed_steps = 0

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        done = done or self._elapsed_steps >= self._max_episode_steps
        if self._elapsed_steps >= self._max_episode_steps:
            info['truncated'] = True
        else:
            info['truncated'] = False
        return obs, reward, done, info


class RewardAtTruncationWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._had_success = False
        # FP dose-response study: inject controlled false positives into the
        # TRAIN reward only (this wrapper is added only when reward_at_truncation
        # is True, i.e. the train env; the eval env uses reward_at_truncation=
        # False so its true-success metric is untouched). With prob
        # FP_INJECT_RATE, a true-failure episode is rewarded as success at
        # truncation — emulating a reward model with a known false-positive rate,
        # to measure how much FP a policy tolerates before it collapses.
        import os
        self._fp_inject = float(os.environ.get("FP_INJECT_RATE", "0.0"))

    def reset(self):
        self._had_success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward >= 1:
            self._had_success = True
        if info['truncated'] and self._had_success:
            rat_reward = 1.0
        elif info['truncated'] and self._fp_inject > 0.0 and np.random.random() < self._fp_inject:
            rat_reward = 1.0          # injected false positive: true failure rewarded as success
            info['fp_injected'] = True
        else:
            rat_reward = 0.0
        return obs, rat_reward, done, info


class PixelMetaWorld:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,  # This measures the number of outer environment steps
        action_repeat,  # Number of inner steps per outer step
        frame_stack,  # Number of outer steps to stack frames over
        obs_stack,  # Number of outer steps to stack obses over
        *,
        reward_shaping=False,
        rl_image_size=None,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_camera=DEFAULT_CAMERA,
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        dual_res=True,
        too_early_reward=False,
        reward_at_truncation=False,
    ):
        assert robots == None or robots == [] or robots == "Sawyer" or robots == ["Sawyer"]
        assert reward_shaping == False, "reward_shaping is not a supported argument"

        assert isinstance(camera_names, list)
        self.camera_names = camera_names

        # Make a state-only environment
        self.env = MetaWorldEnv(
            env_name=env_name,
            camera_name=camera_names[0],
            width=rl_image_size,
            height=rl_image_size,
        )
        # For every outer call to step, make multiple inner calls to step
        self.env = ActionRepeatWrapper(env=self.env, num_repeats=action_repeat)
        # Add a key `prop` to the observation with proprioceptive dimensions
        self.env = ProprioObsWrapper(env=self.env, idx_list=PROP_IDXS[env_name])
        # Add keys to the observation with each camera rendering
        if dual_res:
            self.env = DualResImageObsWrapper(env=self.env, camera_names=camera_names)
        else:
            self.env = ImageObsWrapper(env=self.env, camera_names=camera_names)
        # Add observation stacking for specified number of steps
        self.env = StackObsWrapper(env=self.env, obs_stack=obs_stack, frame_stack=frame_stack)
        # Overwrite environment rewards with sparse rewards
        if too_early_reward:
            self.env = TooEarlySparseRewardWrapper(env=self.env)
        else:
            self.env = SparseRewardWrapper(env=self.env)
        # Set max horizon --> if we get to episode_length steps, done is True
        if episode_length is not None:
            self.env = TimeLimitWrapper(env=self.env, max_episode_steps=episode_length)
            if reward_at_truncation:
                self.env = RewardAtTruncationWrapper(env=self.env)

        self.rl_camera = rl_camera
        self.frame_stack = frame_stack
        # self.image_size = image_size
        self.rl_image_size = rl_image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        # self.resize_transform = None
        # if self.rl_image_size != self.image_size:
        #     self.resize_transform = utils.get_rescale_transform(self.rl_image_size)
        self.num_action = self.env.action_space.shape[0]
        self.observation_shape = (3 * self.frame_stack, rl_image_size, rl_image_size)
        self.state_shape = (STATE_SHAPE[env_name][0] * obs_stack,)
        self.prop_shape = (PROP_SHAPE[env_name][0] * obs_stack,)
        self.device = device
        self.reward_model = None

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.most_recent_info = None

    @property
    def action_dim(self):
        return self.num_action

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def _extract_images(self, obs):
        # NOTE: A couple differences from `_extract_images` in PixelRobosuite:
        # - Logic for adding proprio and image observations is handled by
        #   MetaWorldProprioWrapper and MetaWorldImageWrapper
        # - Logic for frame stacking is handled in StackWrapper

        state = None
        if self.use_state:
            state = torch.from_numpy(obs["state"]).to(self.device)

        prop = torch.from_numpy(obs["prop"]).to(self.device)

        rl_image_obs = None
        all_image_obs = {}
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = torch.from_numpy(obs[image_key].copy())

            highres_key = f"highres_{camera_name}_image"
            if highres_key in obs.keys():
                highres_image_obs = torch.from_numpy(obs[highres_key].copy())
                all_image_obs[camera_name] = highres_image_obs[-3:, :, :]
            else:
                # keep the high-res version for rendering
                # Include just the most recent image if we're using frame stacking
                all_image_obs[camera_name] = image_obs[-3:, :, :]
            if self.rl_camera == camera_name:
                rl_image_obs = image_obs

        assert rl_image_obs is not None
        rl_image_obs = rl_image_obs.to(self.device)
        # if self.resize_transform is not None:
        #     # set the device here because transform is 5x faster on GPU
        #     rl_image_obs = self.resize_transform(rl_image_obs)

        rl_obs = {"obs": rl_image_obs}
        rl_obs["prop"] = prop.to(self.device)

        if self.use_state:
            assert state is not None
            rl_obs["state"] = state.to(self.device)

        return rl_obs, all_image_obs

    def reset(self):
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False

        obs = self.env.reset()
        rl_obs, image_obs = self._extract_images(obs)

        if self.reward_model is not None:
            self.reward_model.reset()

        self.most_recent_info = None

        return rl_obs, image_obs

    def step(self, action):
        self.time_step += 1
        obs, reward, terminal, info = self.env.step(action)
        self.most_recent_info = info

        rl_obs, image_obs = self._extract_images(obs)
        self.episode_reward += reward

        if self.end_on_success and (reward == 1):
            terminal = True
        success = reward == 1

        reward = reward * self.env_reward_scale
        if self.reward_model is not None:
            reward_ret = self.reward_model.get_reward(image_obs)
            reward += reward_ret.reward
            self.episode_extra_reward += reward_ret.reward

        self.terminal = terminal
        return rl_obs, reward, terminal, success, image_obs

    def get_heuristic_action(self, clip_action=False):
        return self.env.get_heuristic_action(clip_action=clip_action)


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = PixelMetaWorld(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        action_repeat=2,
        frame_stack=2,
        obs_stack=1,
        rl_image_size=96,
        device="cpu",
        camera_names=GOOD_CAMERAS["Assembly"],
        use_state=False,
    )
    x = env.reset()[0]["obs"].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x[-3:, :, :], "test_env.png")
