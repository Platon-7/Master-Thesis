# DSRL + VLM Reward Implementation Guide

**Author**: Platon Karageorgis
**Date**: February 9, 2026
**Purpose**: Technical documentation for DSRL training with VLM-based rewards (RoboReward-8B)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Comparison](#architecture-comparison)
3. [File-by-File Implementation](#file-by-file-implementation)
4. [Key Implementation Differences](#key-implementation-differences)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation System](#evaluation-system)
7. [Results and Findings](#results-and-findings)

---

## Overview

This implementation extends **DSRL (Diffusion Steering via Reinforcement Learning)** to use **VLM-based rewards** instead of simulator ground-truth rewards. The goal is to test whether a general-purpose vision-language reward model (RoboReward-8B) can provide sufficient learning signal for policy finetuning on robotic manipulation tasks.

### What is DSRL?

DSRL is a reinforcement learning algorithm that finetunes a pretrained **diffusion policy** using SAC. Instead of learning a policy from scratch, it:
1. Starts with a diffusion policy trained on expert demonstrations
2. Treats noise inputs to the diffusion model as actions
3. Uses SAC to learn how to guide the diffusion process toward higher rewards

### What We Added

We replaced the simulator's sparse binary reward (success=1, failure=0) with **RoboReward-8B's visual assessment** of task progress. The VLM observes episode frames and outputs a 1-5 score indicating task completion.

---

## Architecture Comparison

### Baseline (Ground Truth Rewards)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Environment                      │
│                                                               │
│  Robomimic Lift Env (MuJoCo)                                │
│         ↓                                                     │
│  ObservationWrapperRobomimic (normalize obs)                │
│         ↓                                                     │
│  ActionChunkWrapper (4-step action chunks)                  │
│         ↓                                                     │
│  DummyVecEnv (4 parallel envs)                              │
│         ↓                                                     │
│  DiffusionPolicyEnvWrapper (noise → diffusion → action)     │
│         ↓                                                     │
│  SAC Agent (learns noise policy)                            │
│                                                               │
│  Reward: sim_reward - 1  ∈ {-1, 0}                          │
│    Dense: -1 every step, 0 on success                        │
└─────────────────────────────────────────────────────────────┘
```

### VLM Version (RoboReward-8B Rewards)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Environment                      │
│                                                               │
│  Robomimic Lift Env (MuJoCo)                                │
│         ↓                                                     │
│  VLMRewardWrapperRobomimic (replaces ObservationWrapper)    │
│    • Collects frames every 5 sim steps                       │
│    • At episode end: calls RoboReward-8B                     │
│    • Returns VLM score (1-5) as reward                       │
│         ↓                                                     │
│  ActionChunkWrapper (4-step action chunks)                  │
│         ↓                                                     │
│  DummyVecEnv (4 parallel envs)                              │
│         ↓                                                     │
│  DiffusionPolicyEnvWrapper (noise → diffusion → action)     │
│         ↓                                                     │
│  SAC Agent (learns noise policy)                            │
│                                                               │
│  Reward: VLM score ∈ {1, 2, 3, 4, 5}                        │
│    Sparse: 0 mid-episode, 1-5 at episode end                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            RoboReward-8B (configurable GPU)                   │
│                                                               │
│  Model: Qwen3-VL-8B-Instruct + reward head                │
│  Quantization: 4-bit NF4 (~8GB VRAM)                        │
│  Input: ~32 frames + task instruction                        │
│  Output: Discrete score 1-5                                  │
│    1 = no progress, 5 = task completed                       │
│                                                               │
│  vlm_device: auto → cuda:1 if 2+ GPUs, else cuda:0          │
│  Single A100 80GB fits both policy + VLM                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Architectural Difference**: The VLM reward wrapper **replaces** the observation normalization wrapper and intercepts the reward signal before it reaches the agent.

---

## File-by-File Implementation

### Core Files

#### 1. `train_dsrl_vlm.py` — Main Training Script

**Location**: `dsrl/train_dsrl_vlm.py`

**Purpose**: Entry point for VLM-based DSRL training. Orchestrates model loading, environment creation, and training loop.

**Key Sections**:

```python
# VLM device resolution from config (vlm_device: auto/cuda:0/cuda:1)
vlm_device_cfg = cfg.get('vlm_device', 'auto')
if vlm_device_cfg == 'auto':
    vlm_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
else:
    vlm_device = vlm_device_cfg

roboreward_model = get_roboreward_model(device=vlm_device)
```

**GPU layout**: With 2 GPUs, VLM runs on GPU 1 while SAC trains on GPU 0. With a single large GPU (A100 80GB), both fit on cuda:0 — the policy uses ~2GB and RoboReward-8B 4-bit uses ~8GB.

```python
# Lines 99-119: Training Environment Creation
def make_train_env():
    env = make_vlm_robomimic_env(
        env_name=cfg.env_name,
        roboreward_model=roboreward_model,
        use_vlm_reward=True,  # ← KEY: Enable VLM rewards
        frame_interval=5,      # Capture frame every 5 sim steps
    )
    env = ActionChunkWrapper(env, cfg, max_episode_steps=300)
    return env
```

**Contrast with baseline**: The baseline uses `ObservationWrapperRobomimic` instead of `VLMRewardWrapperRobomimic`. The baseline does NOT load any VLM model.

```python
# Lines 120-142: Evaluation Environment Creation
def make_eval_env():
    env = make_vlm_robomimic_env(
        roboreward_model=roboreward_model,
        use_vlm_reward=False,  # ← IMPORTANT: Eval uses simulator rewards!
    )
    env = ActionChunkWrapper(env, cfg, max_episode_steps=300)
    return env
```

**Why `use_vlm_reward=False` in eval?** We want to measure **actual task success** according to the simulator's ground-truth, not the VLM's possibly inaccurate judgment. The agent learns from VLM rewards but is evaluated on simulator success.

```python
# Lines 144-159: Vectorized Environments
env = make_vec_env(make_train_env, n_envs=4, vec_env_cls=DummyVecEnv)
eval_env = make_vec_env(make_eval_env, n_envs=4, vec_env_cls=DummyVecEnv)

if cfg.algorithm == 'dsrl_sac':
    env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
    eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
```

**Why `DummyVecEnv` instead of `SubprocVecEnv`?** The RoboReward model contains CUDA tensors which cannot be pickled across processes. `DummyVecEnv` runs all 4 environments in the same process sequentially.

**Contrast with baseline**: Uses `SubprocVecEnv` for true parallelism since there's no VLM to share.

**Note**: The unused `SubprocVecEnv` import was removed from `train_dsrl_vlm.py` during cleanup.

```python
# Lines 244-259: VLM-Specific Logging Callback
logging_callback = VLMLoggingCallback(
    eval_env=eval_env,
    eval_freq=70,           # Evaluate every 70 training steps
    eval_episodes=2,        # Run 2 eval episodes per checkpoint
    csv_log_path=csv_log_path,
    vlm_success_threshold=4,  # VLM score ≥ 4 counts as success
)
```

**Contrast with baseline**: Uses `LoggingCallback` (no VLM-specific metrics).

---

#### 2. `vlm_reward_wrapper.py` — VLM Reward Computation

**Location**: `/home/pkarageo/master-thesis/dsrl/vlm_reward_wrapper.py`

**Purpose**: Gym wrapper that intercepts environment steps, collects observation frames, and replaces simulator rewards with VLM scores.

**Class**: `VLMRewardWrapperRobomimic` (lines 19-151)

**Constructor**:
```python
def __init__(self, env, task_name, roboreward_model=None,
             reward_offset=1.0, frame_interval=5, use_vlm_reward=True):
    self.env = env
    self.roboreward_model = roboreward_model
    self.use_vlm_reward = use_vlm_reward  # Train vs eval mode
    self.frame_interval = frame_interval
    self.episode_frames = []        # Accumulates frames during episode
    self.episode_sim_reward = 0.0   # Tracks ground-truth for logging
```

**Reset** (lines 59-70):
```python
def reset(self, **kwargs):
    self.episode_frames = []
    self.step_count = 0
    self.episode_sim_reward = 0.0
    raw_obs = self.env.reset()
    obs = raw_obs['state'].flatten()
    self._capture_frame_from_env()  # Capture initial frame
    return obs
```

**Step — The Core Logic** (lines 72-105):

```python
def step(self, action):
    raw_obs, sim_reward, done, info = self.env.step(action)
    obs = raw_obs['state'].flatten()

    self.step_count += 1
    self.episode_sim_reward += sim_reward  # Track for comparison

    # Collect frames periodically
    if self.use_vlm_reward and self.step_count % self.frame_interval == 0:
        self._capture_frame_from_env()

    # Compute reward based on mode
    if self.use_vlm_reward:
        # TRAINING MODE: Use VLM rewards
        if done:
            self._capture_frame_from_env()  # Final frame
            reward = self._compute_vlm_reward()  # Raw score 1-5
            info['vlm_reward'] = reward
            info['sim_reward'] = self.episode_sim_reward  # Log GT for comparison
            info['sim_success'] = 1 if sim_reward > 0 else 0
        else:
            reward = 0.0  # Sparse: 0 during episode
    else:
        # EVALUATION MODE: Use simulator rewards (ground-truth)
        reward = sim_reward - self.reward_offset
        if done:
            info['sim_reward'] = self.episode_sim_reward
            info['sim_success'] = 1 if self.episode_sim_reward > 0 else 0

    return obs, reward, done, info
```

**Key Points**:
1. **Frame collection**: Only when `use_vlm_reward=True`, captures frames every `frame_interval` steps
2. **Sparse reward**: Outputs `0.0` for all mid-episode steps, then VLM score (1-5) at episode end
3. **Dual logging**: Tracks both VLM reward (what agent sees) and sim reward (ground truth) for analysis
4. **Eval mode**: Falls back to `sim_reward - offset` when `use_vlm_reward=False`

**Contrast with baseline**: `ObservationWrapperRobomimic.step()` is much simpler:
```python
def step(self, action):
    raw_obs, reward, done, info = self.env.step(action)
    reward = reward - self.reward_offset  # Just subtract offset
    obs = raw_obs['state'].flatten()
    return obs, reward, done, info
```
No VLM calls, no frame collection, no info dict augmentation.

**VLM Reward Computation** (lines 116-125):
```python
def _compute_vlm_reward(self) -> float:
    """Compute VLM reward using raw RoboReward progress score {1,...,5}."""
    if self.roboreward_model is None or len(self.episode_frames) == 0:
        return 5.0 if self.episode_sim_reward > 0 else 1.0

    return self.roboreward_model.compute_reward(
        self.episode_frames,
        self.task_instruction,
        return_raw_score=True
    )
```

**Why `return_raw_score=True`?** The RoboReward model has a normalization function that maps [1,5] → [-1,0] (matching the baseline's reward scale). We initially tried this but found it failed completely (see `dsrl_vlm_91522.out` — all metrics went to zero). Using raw [1-5] scores works better because:
- The sparse reward structure is fundamentally different from dense [-1,0]
- Value magnitudes need to be distinguishable from mid-episode zeros
- The wider range [0, 5] vs [0, 0.47] gives SAC's critic meaningful gradients

**Force Episode End** (lines 127-148):
```python
def force_episode_end(self):
    """Called by ActionChunkWrapper when episode hits time limit."""
    if not self.use_vlm_reward:
        return {
            'sim_reward': self.episode_sim_reward,
            'sim_success': 1 if self.episode_sim_reward > 0 else 0,
        }

    self._capture_frame_from_env()
    reward = self._compute_vlm_reward()

    return {
        'vlm_reward': reward,
        'sim_reward': self.episode_sim_reward,
        'sim_success': 1 if self.episode_sim_reward > 0 else 0,
        'num_frames_scored': len(self.episode_frames),
    }
```

**Why is this needed?** Robomimic environments are configured with `"ignore_done": true`, meaning they **never signal episode termination**. Episodes only end when `ActionChunkWrapper` reaches `max_episode_steps`. This method ensures VLM rewards are computed even on truncated episodes.

---

#### 3. `roboreward_wrapper.py` — VLM Model Interface

**Location**: `dsrl/roboreward_wrapper.py`

**Purpose**: Loads and interfaces with the RoboReward-8B model. Handles frame preprocessing, prompt construction, and score parsing.

**Class**: `RoboRewardModel`

**Constructor**:
```python
def __init__(self, model_path=None, device="cuda:0", dtype=torch.float16, max_frames=32):
    # Auto-detect model path from HF cache (portable)
    if model_path is None:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        base_cache = os.path.join(hf_home, "models--teetone--RoboReward-8B", "snapshots")
        # ... find latest snapshot

    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Place entire model on specified GPU
    gpu_id = int(device.split(":")[1])
    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": gpu_id},
    )
    self.processor = AutoProcessor.from_pretrained(model_path)
```

**Model path auto-detection**: Uses the `HF_HOME` environment variable (standard HuggingFace convention) to locate cached model weights. No hardcoded paths — works on any cluster where the model has been downloaded.

**Default device is `cuda:0`**: Supports single-GPU setups out of the box. The `vlm_device` config in the YAML (passed via `get_roboreward_model(device=...)`) overrides this for multi-GPU setups.

**Why 4-bit quantization?** RoboReward-8B is based on Qwen3-VL-7B (~14GB in FP16). 4-bit NF4 quantization reduces this to ~8GB with minimal accuracy loss, allowing it to fit on a 24GB GPU alongside framebuffer memory.

**Compute Reward — Main API** (lines 103-223):
```python
def compute_reward(self, frames, task_instruction, return_raw_score=False):
    # Frame subsampling if > max_frames
    if len(frames) > self.max_frames:
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Convert frames to PIL Images
    images = [Image.fromarray(f) for f in frames]

    # Construct prompt
    prompt = self._build_prompt(task_instruction, num_frames=len(frames))

    # Format messages for Qwen3VL
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img} for img in images
    ] + [{"type": "text", "text": prompt}]}]

    # Tokenize and generate
    text = self.processor.apply_chat_template(messages, tokenize=False)
    inputs = self.processor(
        text=[text], images=images, return_tensors="pt"
    ).to(self.device)

    generated_ids = self.model.generate(**inputs, max_new_tokens=128)
    output_text = self.processor.batch_decode(generated_ids)[0]

    # Parse score from "ANSWER: X" format
    score_match = re.search(r'ANSWER:\s*([1-5])', output_text)
    raw_score = int(score_match.group(1)) if score_match else 1

    if return_raw_score:
        return float(raw_score)

    # Normalize to [-1, 0] (NOT USED in current implementation)
    normalized_reward = (raw_score - 5) / 4.0  # Maps [1,5] → [-1,0]
    return normalized_reward
```

**Frame Subsampling**: If an episode generates >32 frames (e.g., 300 sim steps / 5 = 60 frames), uniformly subsample to 32. This keeps VLM inference time constant and fits within the model's context window.

**Prompt Construction** (lines 125-158):
```python
def _build_prompt(self, task_instruction, num_frames):
    return f"""You are evaluating a robot manipulation task.

Task: {task_instruction}

You are given {num_frames} frames showing the robot's attempt.

Rate the progress on a scale of 1-5:
1: No progress / robot didn't move
2: Robot moved but in wrong direction or knocked object over
3: Partial progress (e.g., grasped object but didn't lift)
4: Near completion (e.g., lifted object but not to target)
5: Task fully completed

Output ONLY the number in the format:
ANSWER: X

where X is 1, 2, 3, 4, or 5."""
```

**Why this format?** The "ANSWER: X" format ensures deterministic parsing via regex. Free-form responses are harder to parse reliably.

**Task Instructions** (lines 28-33):
```python
TASK_INSTRUCTIONS = {
    'lift': 'Pick up the red cube from the table',
    'can': 'Pick the can and place it into the bin',
    'square': 'Pick the square nut and place it onto the peg',
    'transport': 'Pick the object and move it to the target location',
}
```

These map Robomimic environment names to natural language task descriptions for the VLM.

---

#### 4. `env_utils.py` — Action Chunking Wrapper

**Location**: `/home/pkarageo/master-thesis/dsrl/env_utils.py`

**Purpose**: Chunks 4 consecutive low-level actions into one high-level decision. Critical for proper integration with VLM wrapper.

**Class**: `ActionChunkWrapper` (lines 144-218)

**Step Method** (lines 161-212):
```python
def step(self, action):
    if len(action.shape) == 1:
        action = action.reshape(self.act_steps, -1)  # Reshape to (4, action_dim)

    obs_, reward_, done_, info_ = [], [], [], []
    done_i = False
    episode_end_info = None

    # Execute 4 primitive actions
    for i in range(action.shape[0]):
        self.count += 1
        obs_i, reward_i, done_i, info_i = self.env.step(action[i])
        obs_.append(obs_i)
        reward_.append(reward_i)
        done_.append(done_i)
        info_.append(info_i)

        # Preserve info from the step where episode ended
        if done_i and episode_end_info is None:
            episode_end_info = info_i.copy()

    obs = obs_[-1]  # Use final observation
    reward = sum(reward_)  # Sum rewards over chunk
    done = np.max(done_)
    info = info_[-1]

    # Merge episode end info into final info
    if episode_end_info is not None:
        for key in ['vlm_reward', 'sim_reward',
                    'sim_success', 'num_frames_scored']:
            if key in episode_end_info:
                info[key] = episode_end_info[key]

    # Handle TimeLimit truncation
    truncated = False
    if self.count >= self.max_episode_steps:
        if not done:
            truncated = True
            # Force VLM reward computation
            if hasattr(self.env, 'force_episode_end'):
                vlm_info = self.env.force_episode_end()
                info.update(vlm_info)
                if 'vlm_reward' in vlm_info:
                    reward = vlm_info['vlm_reward']  # Override summed reward
        done = True

    if done:
        info['terminal_observation'] = obs
        info['TimeLimit.truncated'] = truncated

    return obs, reward, done, truncated, info
```

**Key Logic**:
1. **Info preservation** (lines 177-189): If the inner env signals `done=True` on step 2 of the chunk, we preserve that step's info dict (which contains VLM reward) and merge it into the final info
2. **Truncation handling** (lines 195-204): When time limit is reached, calls `force_episode_end()` to ensure VLM reward is computed even if inner env didn't signal done
3. **Reward override** (line 204): The VLM reward from truncation replaces the summed chunk reward (which would just be 0+0+0+0 in sparse mode)

**Contrast with baseline**: The baseline's `ActionChunkWrapper` is identical except it doesn't have the `force_episode_end()` logic (since no VLM is involved).

---

#### 5. `utils.py` — VLM Logging Callback

**Location**: `/home/pkarageo/master-thesis/dsrl/utils.py`

**Purpose**: Custom SB3 callback for logging VLM-specific metrics to WandB and CSV.

**Class**: `VLMLoggingCallback` (lines 166-450)

**Tracked Metrics** (lines 214-218):
```python
self.episode_vlm_rewards = []    # VLM scores (1-5)
self.episode_sim_rewards = []    # Ground-truth cumulative rewards
self.episode_sim_success = []    # Binary success (sim_reward > 0)
self.total_timesteps = 0         # Cumulative RL steps
```

**On Step — Training Metrics** (lines 269-355):
```python
def _on_step(self):
    # Collect episode-end info from parallel envs
    for info in self.locals['infos']:
        if 'vlm_reward' in info:
            self.episode_vlm_rewards.append(info['vlm_reward'])
            if 'sim_reward' in info:
                self.episode_sim_rewards.append(info['sim_reward'])
                self.episode_sim_success.append(1 if info['sim_reward'] > 0 else 0)

    self.total_timesteps += self.action_chunk * self.model.n_envs

    # Log every log_freq steps (default: 75 = 1 episode per env)
    if self.n_calls % self.log_freq == 0 and len(self.episode_vlm_rewards) > 0:
        mean_vlm_reward = np.mean(self.episode_vlm_rewards)
        sim_success_rate = np.mean(self.episode_sim_success)
        vlm_success_rate = np.mean([1 if s >= self.vlm_success_threshold else 0
                                   for s in self.episode_vlm_rewards])

        # WandB logging
        wandb.log({
            "vlm/reward": mean_vlm_reward,
            "vlm/success_rate": vlm_success_rate,
            "sim/success_rate": sim_success_rate,
            "timestep": self.total_timesteps,
        }, step=self.total_timesteps)

        print(f"[Step {self.total_timesteps}] VLM Reward: {mean_vlm_reward:.2f}, "
              f"VLM Success: {vlm_success_rate*100:.1f}%, "
              f"Sim Success: {sim_success_rate*100:.1f}%")

    # Evaluation every eval_freq steps
    if self.n_calls % self.eval_freq == 0:
        self.evaluate(self.locals['self'], deterministic=False)
```

**Evaluation** (`VLMLoggingCallback.evaluate()`):
```python
def evaluate(self, agent, deterministic=False):
    env = self.eval_env
    eval_vlm_rewards, eval_sim_success, eval_episode_rewards = [], [], []

    with torch.no_grad():
        for i in range(self.eval_episodes):
            obs = env.reset()
            episode_done = np.zeros(obs.shape[0], dtype=bool)
            episode_rewards = np.zeros(obs.shape[0])

            for step_idx in range(self.max_steps):
                action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                next_obs, reward, done, infos = env.step(action)
                obs = next_obs

                # Accumulate rewards for non-done envs
                for j in range(len(done)):
                    if not episode_done[j]:
                        episode_rewards[j] += reward[j]

                # Collect metrics from completed episodes
                for j, (d, info) in enumerate(zip(done, infos)):
                    if d and not episode_done[j]:
                        episode_done[j] = True
                        eval_episode_rewards.append(episode_rewards[j])
                        if 'vlm_reward' in info:
                            eval_vlm_rewards.append(info['vlm_reward'])
                        if 'sim_success' in info:
                            eval_sim_success.append(info['sim_success'])
                        elif 'sim_reward' in info:
                            eval_sim_success.append(1 if info['sim_reward'] > 0 else 0)

                # Early termination: stop once all envs have completed
                if np.all(episode_done):
                    break

            # Timeout fallback: count unfinished envs as failures
            for j in range(obs.shape[0]):
                if not episode_done[j]:
                    eval_sim_success.append(0)
                    eval_episode_rewards.append(episode_rewards[j])
```

**Early termination**: Once all parallel eval envs have completed their episode (success or failure), the inner loop breaks immediately instead of stepping through the remaining `max_steps`. This avoids wasting compute on already-finished episodes. Metrics are recorded at the moment each env finishes (before the break), so logging is unaffected. Envs that never finish within `max_steps` are counted as failures by the timeout fallback.

**Note**: This early termination is only in `VLMLoggingCallback.evaluate()`, not in `LoggingCallback.evaluate()` (GT baseline). The GT callback uses `SubprocVecEnv` with 25 eval envs and auto-reset semantics where multiple episodes can complete per inner loop — different accounting that doesn't benefit from early breaking.

**Key Features**:
- **Dual metrics**: Tracks both VLM judgments (what agent learns from) and simulator truth (what we care about)
- **Success threshold**: VLM score >= 4 counts as success (scores 4 and 5 are "near completion" and "fully completed")
- **CSV logging**: All metrics saved to timestamped CSV for offline analysis
- **WandB alignment**: Uses `step=self.total_timesteps` consistently so training and eval curves share x-axis
- **Early eval termination**: Breaks inner loop when all envs are done, saving unnecessary stepping

**Contrast with baseline**: `LoggingCallback` only tracks `sim_reward` and `episode_length`, no VLM metrics.

---

## Key Implementation Differences

### Summary Table

| Aspect | Baseline (GT) | VLM Version |
|--------|--------------|-------------|
| **Reward Source** | Simulator | RoboReward-8B VLM |
| **Reward Structure** | Dense: -1 per step, 0 on success | Sparse: 0 mid-episode, 1-5 at end |
| **Reward Range** | {-1, 0} | {0, 1, 2, 3, 4, 5} |
| **Wrapper** | `ObservationWrapperRobomimic` | `VLMRewardWrapperRobomimic` |
| **Frame Collection** | None | Every 5 sim steps, ~32 frames/episode |
| **VLM Model** | N/A | Qwen3-VL-7B (4-bit), GPU 1 |
| **VLM Calls** | 0 | 1 per episode (at termination) |
| **Vec Env** | `SubprocVecEnv` (parallel) | `DummyVecEnv` (sequential, CUDA constraint) |
| **Logging Callback** | `LoggingCallback` | `VLMLoggingCallback` |
| **Evaluation** | Sim rewards, runs full episode | Sim rewards, early termination on done |
| **Training Script** | `train_dsrl.py` | `train_dsrl_vlm.py` |
| **Configs** | `dsrl_{lift,can,square}.yaml` | `dsrl_{lift,can,square}_vlm.yaml` |
| **GPU Layout** | Single GPU | Configurable: `vlm_device: auto` (1 or 2 GPUs) |

### Critical Differences Explained

#### 1. Reward Signal Shape

**Baseline**: Dense negative reward creates a "penalty world" where the agent is punished every step and learns to reach states where punishment stops.

**VLM**: Sparse positive reward creates a "goal world" where the agent explores freely mid-episode and receives delayed feedback at the end. This requires the critic to learn long-horizon value estimates.

**Implication**: The SAC hyperparameters (entropy coefficient, learning rate, target entropy) were tuned for the dense baseline. The sparse VLM signal operates in a fundamentally different regime.

#### 2. Reward Scale

**Baseline**: Range of 1 (from -1 to 0). Over a 75-step episode, cumulative return ranges from -75 to 0.

**VLM**: Range of 5 (from 1 to 5). Over a 75-step episode, terminal return is 1-5, discounted to ~0.47-2.35 at episode start (with γ=0.99).

**Why not normalize?** We tried mapping [1,5] → [-1,0] to match the baseline (see `dsrl_vlm_91522.out`). Result: **total training collapse**. The critic learned "everything ≈ 0" because the best outcome (score 5 → reward 0) was indistinguishable from mid-episode reward (0). Entropy coefficient collapsed to ~10^-6 and sim success stayed at 0%.

Using raw [1,5] scores, the critic has a 5x wider value range to learn from, preventing collapse.

#### 3. VLM Inference Overhead

**Per-episode cost**: ~0.5-1 second for VLM inference (32 frames, 4-bit quantized model)

**Training speed**:
- Baseline: ~156K steps/day on 1 GPU
- VLM: ~50-70K steps/day on 2 GPUs (1 for SAC, 1 for VLM)

The VLM itself is fast (~0.5s), but the sequential `DummyVecEnv` slows overall throughput. The baseline's `SubprocVecEnv` runs 4 MuJoCo sims in parallel, while VLM version runs them sequentially.

#### 4. Frame Collection Strategy

**Where**: `VLMRewardWrapperRobomimic`, every 5 simulator steps
**Why every 5?** Trade-off between:
- More frames = better VLM understanding of dynamics
- Fewer frames = faster inference, less memory

For a 300-step episode: 300/5 = 60 frames → subsampled to 32 frames (max context).

**Rendering**: Uses MuJoCo's offscreen rendering (`render_offscreen=True`) in 84x84 RGB. No on-screen display needed.

---

## Training Pipeline

### 1. Initialization Phase

```
1. Resolve vlm_device from config ("auto" → cuda:1 if 2+ GPUs, else cuda:0)
2. Load RoboReward-8B onto vlm_device (4-bit quantized, ~8GB VRAM)
   Model path auto-detected from $HF_HOME cache
3. Load pretrained diffusion policy (from robomimic checkpoint)
4. Create 4 training envs (DummyVecEnv)
   └─ Each env: Robomimic → VLMRewardWrapper → ActionChunkWrapper
5. Create 4 eval envs (DummyVecEnv)
   └─ Each env: Robomimic → VLMRewardWrapper(use_vlm_reward=False) → ActionChunkWrapper
6. Wrap vectorized envs in DiffusionPolicyEnvWrapper
7. Initialize SAC agent on cfg.device (default cuda:0)
8. Initialize replay buffer with offline demo data (optional)
9. Run initial exploration rollouts (config-dependent, e.g. 24000 for lift)
```

### 2. Training Loop (1 iteration = 1 RL step)

```
For each training step:
  1. SAC agent samples noise action from policy
  2. DiffusionPolicyEnvWrapper: noise → diffusion policy → 4-step action chunk
  3. ActionChunkWrapper executes 4 primitive actions in MuJoCo

  For each primitive action:
    a. MuJoCo simulation step
    b. VLMRewardWrapper observes sim_reward
    c. If step_count % 5 == 0: capture frame (84x84 RGB)
    d. If done:
       i.  Capture final frame
       ii. Call RoboReward-8B with all frames → score (1-5)
       iii. Return vlm_reward as reward signal
    e. Else: return reward=0.0

  4. Accumulate (s, a, r, s', done) in replay buffer
  5. Every step, run 30 gradient updates (UTD=30):
     - Sample batch from replay buffer
     - Update SAC critic (Q-functions)
     - Update SAC actor (policy)
     - Update noise critic (DSRL-specific)
```

**UTD=30**: The agent performs 30 gradient updates per 1 environment step. This is critical for sample efficiency with sparse rewards — the agent needs to squeeze as much learning as possible from each episode.

### 3. Logging (every 75 steps = ~1 episode per env)

```
Collect from last 75 steps:
  - VLM scores (from 'vlm_reward' in info)
  - Sim success flags (from 'sim_success' in info)

Compute:
  - mean_vlm_score = average of raw VLM scores
  - vlm_success_rate = fraction of episodes with VLM score ≥ 4
  - sim_success_rate = fraction of episodes with sim_reward > 0

Log to:
  - WandB: vlm/reward, vlm/success_rate, sim/success_rate
  - CSV: timestamped row with all metrics
  - Stdout: "[Step X] VLM Score: ..., VLM Success: ..., Sim Success: ..."
```

### 4. Evaluation (every eval_freq steps)

```
For each batch (eval_episodes iterations, 4 envs per batch):
  1. Reset eval_env (use_vlm_reward=False → simulator rewards)
  2. Step policy until all 4 envs done OR max_steps reached
     - Record sim_success + episode_reward when each env finishes
     - Break early once all envs are done (no wasted compute)
  3. Any env not done by max_steps → counted as failure
  4. Accumulate results across all batches
  5. Log to WandB: eval/sim_success_rate, eval/episode_reward
  6. Print: "[Eval @ step X] Sim Success: Y%"
```

**Early termination**: The inner step loop breaks as soon as all parallel envs have completed their episode. Since VLM eval is sequential (DummyVecEnv), this avoids stepping through hundreds of unnecessary steps after all envs are done.

**Why sim rewards in eval?** The VLM might hallucinate success. We care about actual task completion according to the simulator's ground-truth success condition.

---

## Evaluation System

### Two-Level Evaluation

#### Training Metrics (VLM-based)
- **What**: VLM scores from training rollouts
- **Frequency**: Logged every 75 steps
- **Purpose**: Shows what reward signal the agent is learning from
- **Metrics**: `vlm/reward`, `vlm/success_rate`, `sim/success_rate`

#### Evaluation Metrics (Simulator-based)
- **What**: Dedicated eval episodes with simulator rewards
- **Frequency**: Every 70 steps, 2 episodes per eval
- **Purpose**: Trustworthy measure of actual task performance
- **Metrics**: `eval/sim_success_rate`, `eval/episode_reward`

### Evaluation Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Evaluation Trigger (every eval_freq steps)   │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  For i in range(eval_episodes):  # e.g. 50 batches      │
│    Reset Eval Env (4 parallel envs)                     │
│      use_vlm_reward=False → uses simulator rewards      │
│                                                          │
│    For step in range(max_steps):                        │
│      1. Step all envs with policy action                │
│      2. For each env that finishes (done=True):         │
│         - Record sim_success, episode_reward            │
│         - Mark env as episode_done                      │
│      3. If ALL envs done → break early                  │
│                                                          │
│    Any env still not done → count as failure (success=0) │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Aggregate Results (4 envs × 50 batches = 200 episodes)  │
│    eval_sim_success_rate = mean(sim_success_list)       │
│    eval_episode_reward = mean(episode_reward_list)      │
│    eval_vlm_reward = mean(vlm_rewards) if available     │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Log to WandB & CSV                                      │
│    eval/sim_success_rate, eval/episode_reward            │
│    eval/vlm_reward, eval/vlm_success_rate               │
│    timestep (for x-axis alignment)                       │
└─────────────────────────────────────────────────────────┘
```

The early termination (`break` when all envs done) means evaluation batches where all 4 envs finish quickly skip the remaining steps. This saves significant time since each step involves a diffusion policy forward pass. Metrics are identical to running the full `max_steps` because each env's results are recorded at its moment of completion.

### Recent Fix (Feb 9, 2026)

**Problem**: Eval metrics were not appearing in WandB or logs.

**Root Cause**:
1. Robomimic envs have `"ignore_done": true` → never signal natural termination
2. Episodes only end via ActionChunkWrapper's truncation
3. `VLMRewardWrapperRobomimic.force_episode_end()` returned `{}` when `use_vlm_reward=False`
4. Eval envs never had `sim_success` in their info dicts

**Fix**: Modified `force_episode_end()` to return sim metrics even in eval mode:
```python
if not self.use_vlm_reward:
    return {
        'sim_reward': self.episode_sim_reward,
        'sim_success': 1 if self.episode_sim_reward > 0 else 0,
    }
```

Also aligned WandB x-axis (changed from `step=log_count` to `step=total_timesteps`) and improved CSV logging.

### Portability & Multi-Task Update (Feb 17, 2026)

**Changes for Snellius cluster portability and multi-task support:**

1. **Portable paths**: Replaced all hardcoded `/var/scratch/pkarageo/` paths with environment variables:
   - `roboreward_wrapper.py`: Model path auto-detected from `$HF_HOME` (standard HuggingFace convention)
   - `dsrl_lift_vlm.yaml`: `log_dir` changed from hardcoded scratch path to `./logs` (overridable via CLI)
   - Job scripts: Use `$SCRATCH`, `$HOME`, `$WANDB_API_KEY` instead of hardcoded values

2. **Configurable VLM device**: Added `vlm_device: auto` config key. Resolves to `cuda:1` if 2+ GPUs available, else `cuda:0`. Supports single-GPU setups (A100 80GB fits both policy + VLM).

3. **Multi-task VLM configs**: Created `dsrl_can_vlm.yaml` and `dsrl_square_vlm.yaml` with task-specific hyperparameters (obs_dim, network architecture, discount, init_rollout_steps) inherited from their GT counterparts.

4. **Early eval termination**: `VLMLoggingCallback.evaluate()` now breaks the inner step loop once all parallel eval envs have completed their episode, avoiding wasted compute. Metrics are recorded at each env's completion moment, so logging is unaffected.

5. **Code cleanup**: Removed bare `except: pass` blocks, unused `SubprocVecEnv` import, emoji characters from prints, and development comment markers (`# NEW:`, `# MODIFIED:`). Added `ZeroDivisionError` guard in `LoggingCallback._on_step()`.

6. **Security**: Removed hardcoded WandB API key from DAS-6 job scripts, replaced with `$WANDB_API_KEY` environment variable reference.

7. **Snellius job scripts**: Created portable `jobs/snellius_vlm_train.job` and `jobs/snellius_gt_train.job` using Snellius conventions (`$SCRATCH`, `gpu_a100` partition, `module load 2024`).

---

## Results and Findings

### Training Run: `dsrl_vlm_91539`

**Setup**:
- Environment: Robomimic Lift (pick up red cube)
- Task instruction: "Pick up the red cube from the table"
- Training budget: 250K steps (~3,332 episodes)
- Duration: ~48 hours (Feb 7-9, 2026)
- Hardware: 2× NVIDIA Titan RTX (24GB each)

**Hyperparameters**:
```yaml
algorithm: dsrl_na
total_timesteps: 250000
n_envs: 4
max_episode_steps: 300
act_steps: 4  # Action chunking

train:
  utd: 30  # Update-to-data ratio
  action_magnitude: 1.5  # Noise action space
  layer_size: 2048
  learning_rate: 0.0003
  gamma: 0.99
  init_rollout_steps: 1501
```

### Quantitative Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Final `ep_rew_mean`** | 2.7–3.2 | VLM consistently scores episodes around 3/5 |
| **VLM Success Rate** | 34.9% | VLM judges 35% of episodes as successful (score ≥ 4) |
| **Sim Success Rate** | 45.0% | Simulator reports 45% actual success |
| **VLM-Sim Agreement** | 47.4% | VLM and simulator agree less than coin flip |
| **Pearson Correlation** | -0.014 | Essentially zero correlation |
| **Confusion Matrix** | | |
| - True Positives (both success) | 25.2% | VLM correctly identifies success |
| - True Negatives (both fail) | 22.2% | VLM correctly identifies failure |
| - **False Negatives (VLM misses success)** | **34.3%** | Largest error mode! |
| - False Positives (VLM hallucinates) | 18.2% | VLM false alarms |

### Training Curves

**`ep_rew_mean`** (VLM score):
- Starts at ~4.25 (likely base policy doing reasonably well)
- Drops to ~2.7 by step 50K (exploration degrades performance)
- Oscillates between 2.6-3.3 for remainder of training
- **No upward trend** — agent is not improving according to VLM

**Losses**:
- `critic_loss`: Stable at ~0.007 (very low due to sparse small-magnitude rewards)
- `actor_loss`: Stable at ~-2.6
- `ent_coef`: Stable at ~0.001 (entropy coefficient)

**Comparison to Baseline (GT)**:
- GT `ep_rew_mean`: -264 → -100 (clear improvement)
- GT `critic_loss`: ~3 (higher due to larger Q-value magnitudes)
- GT `actor_loss`: Rises to ~50 (different scale due to reward structure)

### Key Finding: VLM Reward Quality is Insufficient

The **primary bottleneck is RoboReward-8B's accuracy**, not the DSRL algorithm or implementation. Evidence:

1. **Poor VLM-Sim correlation** (-0.014): The VLM's judgments are uncorrelated with actual task success
2. **High false negative rate** (34.3%): The VLM frequently scores successful episodes as 2/5 or 3/5, punishing correct behavior
3. **No learning trend**: `ep_rew_mean` stays flat while baseline GT steadily improves
4. **Sim success better than VLM suggests**: Episodes with sim success 75-100% often get VLM scores of 2-3/5

The agent extracts what signal it can (losses converge, metrics stabilize), but the noisy VLM feedback prevents meaningful policy improvement.

### Probable Cause

RoboReward-8B is a **general-purpose** reward model trained on a broad dataset. For the specific Lift task:
- It may lack fine-grained understanding of "grasp → lift → hold" sequence
- Static frames every 5 steps may miss critical moments (grasp contact, lift initiation)
- The 1-5 rubric may not align well with the task's success criteria

The RoboReward paper likely used a **task-specific finetuned VLM** (Qwen2.5-VL finetuned on Lift data) for their Figure 2 results, not the generic RoboReward-8B model.

---

## Next Steps

### For Reproducing Paper Results
1. **Obtain task-specific VLM**: Finetune Qwen3-VL on Lift success/failure data, or use RoboReward's smaller 3B model if it's task-specific
2. **Increase VLM calls**: Try per-step VLM rewards instead of per-episode (expensive but denser signal)
3. **Improve frame selection**: Use keyframe detection or active learning to capture task-critical moments

### For Improving VLM Reward Models (Thesis Direction)
1. **Analyze failure modes**: Visualize episodes where VLM and sim disagree — what is the VLM seeing wrong?
2. **Reward model ablation**: Test different VLM backbones (Qwen3-VL-2B vs 7B vs 32B)
3. **Prompt engineering**: Experiment with chain-of-thought reasoning or multi-turn dialogue
4. **Hybrid rewards**: Combine VLM scores with auxiliary signals (gripper closure, object velocity)

### For Extending to Other Tasks
The implementation is task-agnostic. VLM configs already exist for **lift**, **can**, and **square**:
- `cfg/robomimic/dsrl_lift_vlm.yaml` (obs_dim=19, max_steps=300)
- `cfg/robomimic/dsrl_can_vlm.yaml` (obs_dim=23, max_steps=300)
- `cfg/robomimic/dsrl_square_vlm.yaml` (obs_dim=23, max_steps=400, larger network)

Each config inherits the task-specific hyperparameters from its GT counterpart (base policy path, network architecture, discount, UTD) and adds VLM settings (`vlm_device`, `use_vlm_reward`). Task instructions are defined in `roboreward_wrapper.py:TASK_INSTRUCTIONS`.

To add a new task (e.g., transport): copy an existing VLM yaml, update `env_name`, `obs_dim`, `base_policy_path`, and add the task instruction to `TASK_INSTRUCTIONS`.

---

## Conclusion

This implementation successfully integrates VLM-based rewards into DSRL, demonstrating:

✅ **Correct VLM integration**: Frame collection, reward substitution, dual logging all work
✅ **Proper evaluation**: Separate eval with sim rewards provides ground-truth metrics
✅ **Stable training**: Losses converge, no crashes or OOM errors
✅ **Reproducible pipeline**: Configs, logging, checkpointing all in place

❌ **VLM reward quality insufficient**: RoboReward-8B cannot reliably judge Lift task success
❌ **No policy improvement**: Flat training curves indicate learning is bottlenecked by noisy reward

The implementation is sound. The challenge is VLM accuracy — which is precisely the research question for improving vision-language reward models.

---

## File Reference Summary

| File | Purpose |
|------|---------|
| `train_dsrl_vlm.py` | Main VLM training script, orchestration |
| `vlm_reward_wrapper.py` | VLM reward computation wrapper |
| `roboreward_wrapper.py` | RoboReward-8B model interface (portable paths) |
| `utils.py` | LoggingCallback + VLMLoggingCallback (eval with early termination) |
| `env_utils.py` | ActionChunkWrapper, truncation handling |
| `cfg/robomimic/dsrl_lift_vlm.yaml` | Lift VLM config (obs_dim=19) |
| `cfg/robomimic/dsrl_can_vlm.yaml` | Can VLM config (obs_dim=23) |
| `cfg/robomimic/dsrl_square_vlm.yaml` | Square VLM config (obs_dim=23, larger network) |
| `jobs/snellius_vlm_train.job` | Portable Snellius job script (VLM) |
| `jobs/snellius_gt_train.job` | Portable Snellius job script (GT) |

---

**For questions or debugging, refer to**:
- WandB dashboard: `https://wandb.ai/nlp-squad/dsrl_roboreward`
- Training logs: `dsrl/dsrl_vlm_*.out`
- Configs: `dsrl/cfg/robomimic/dsrl_{lift,can,square}_vlm.yaml`
