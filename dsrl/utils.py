import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import csv
import os
from datetime import datetime


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / max(np.sum(self.episode_completed), 1),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps


class VLMLoggingCallback(BaseCallback):
	"""
	Callback specifically designed for VLM reward training.
	Logs both simulator success and VLM scores for proper plotting.
	
	Logged metrics for plotting:
	- vlm/reward: Mean VLM reward per episode [1-5] (raw discrete progress score)
	- vlm/success_rate: Success rate based on VLM reward >= threshold
	- sim/success_rate: Success rate based on simulator reward
	- eval/vlm_reward: Mean VLM reward during evaluation
	- eval/sim_success_rate: Simulator success rate during evaluation
	- eval/vlm_success_rate: VLM-based success rate during evaluation
	"""
	
	def __init__(
		self,
		action_chunk=4,
		log_freq=1000,
		use_wandb=True,
		eval_env=None,
		eval_freq=70,
		eval_episodes=2,
		verbose=0,
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
		csv_log_path=None,  # Path to save CSV logs
		vlm_success_threshold=4,  # VLM score >= this = success
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.csv_log_path = csv_log_path
		self.vlm_success_threshold = vlm_success_threshold
		
		# VLM-specific tracking
		self.episode_vlm_rewards = []
		self.episode_sim_rewards = []
		self.episode_sim_success = []  # Track simulator success per episode
		self.total_timesteps = 0
		
		# CSV logging setup
		self.csv_data = []
		if csv_log_path:
			self._setup_csv(csv_log_path)
	
	def _setup_csv(self, path):
		"""Setup CSV file with headers."""
		os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
		with open(path, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(self._csv_columns())

	@staticmethod
	def _csv_columns():
		return [
			'timestep', 'log_step', 'datetime', 'source',
			'train_vlm_reward',
			'train_sim_success_rate', 'train_vlm_success_rate',
			'train_actor_loss', 'train_critic_loss', 'train_ent_coef',
			'eval_sim_success_rate', 'eval_episode_reward',
			'eval_vlm_reward', 'eval_vlm_success_rate',
		]

	def _log_to_csv(self, data_dict):
		"""Append data to CSV file."""
		if self.csv_log_path:
			# Determine if this is a train or eval row
			source = 'eval' if 'eval_sim_success_rate' in data_dict and 'train_vlm_reward' not in data_dict else 'train'
			with open(self.csv_log_path, 'a', newline='') as f:
				writer = csv.writer(f)
				writer.writerow([
					data_dict.get('timestep', ''),
					data_dict.get('log_step', ''),
					datetime.now().isoformat(),
					source,
					data_dict.get('train_vlm_reward', ''),
					data_dict.get('train_sim_success_rate', ''),
					data_dict.get('train_vlm_success_rate', ''),
					data_dict.get('train_actor_loss', ''),
					data_dict.get('train_critic_loss', ''),
					data_dict.get('train_ent_coef', ''),
					data_dict.get('eval_sim_success_rate', ''),
					data_dict.get('eval_episode_reward', ''),
					data_dict.get('eval_vlm_reward', ''),
					data_dict.get('eval_vlm_success_rate', ''),
				])
	
	def _on_step(self):
		# Extract info from episode ends
		for info in self.locals['infos']:
			if 'vlm_reward' in info:
				# Episode ended with VLM reward
				self.episode_vlm_rewards.append(info['vlm_reward'])
				if 'sim_reward' in info:
					self.episode_sim_rewards.append(info['sim_reward'])
					# Sim success: positive cumulative reward
					self.episode_sim_success.append(1 if info['sim_reward'] > 0 else 0)
		
		self.total_timesteps += self.action_chunk * self.model.n_envs
		
		# Log losses more frequently (every 100 steps) for smooth curves
		if self.use_wandb and self.n_calls % 100 == 0:
			try:
				logger = self.locals['self'].logger
				actor_loss = logger.name_to_value.get('train/actor_loss', None)
				critic_loss = logger.name_to_value.get('train/critic_loss', None)
				noise_critic_loss = logger.name_to_value.get('train/noise_critic_loss', None)
				ent_coef = logger.name_to_value.get('train/ent_coef', None)
				
				loss_metrics = {"timestep": self.total_timesteps}
				if actor_loss is not None:
					loss_metrics["loss/actor"] = actor_loss
				if critic_loss is not None:
					loss_metrics["loss/critic"] = critic_loss
				if noise_critic_loss is not None:
					loss_metrics["loss/noise_critic"] = noise_critic_loss
				if ent_coef is not None:
					loss_metrics["loss/ent_coef"] = ent_coef
				
				if len(loss_metrics) > 1:  # More than just timestep
					wandb.log(loss_metrics, step=self.total_timesteps)
			except Exception as e:
				print(f"Warning: could not log losses: {e}")
		
		# Log at intervals
		if self.n_calls % self.log_freq == 0 and len(self.episode_vlm_rewards) > 0:
			self.log_count += 1
			
			# Compute metrics
			mean_vlm_reward = np.mean(self.episode_vlm_rewards)
			sim_success_rate = np.mean(self.episode_sim_success) if self.episode_sim_success else 0
			# VLM success: score >= threshold
			vlm_success_rate = np.mean([1 if s >= self.vlm_success_threshold else 0
									   for s in self.episode_vlm_rewards])
			
			# Get loss values
			try:
				actor_loss = self.locals['self'].logger.name_to_value.get('train/actor_loss', 0)
				critic_loss = self.locals['self'].logger.name_to_value.get('train/critic_loss', 0)
				ent_coef = self.locals['self'].logger.name_to_value.get('train/ent_coef', 0)
			except Exception:
				actor_loss, critic_loss, ent_coef = 0, 0, 0
			
			csv_data = {
				'timestep': self.total_timesteps,
				'log_step': self.log_count,
				'train_vlm_reward': mean_vlm_reward,
				'train_sim_success_rate': sim_success_rate,
				'train_vlm_success_rate': vlm_success_rate,
				'train_actor_loss': actor_loss,
				'train_critic_loss': critic_loss,
				'train_ent_coef': ent_coef,
			}
			
			if self.use_wandb:
				wandb.log({
					"vlm/reward": mean_vlm_reward,
					"vlm/success_rate": vlm_success_rate,
					"sim/success_rate": sim_success_rate,
					"timestep": self.total_timesteps,
				}, step=self.total_timesteps)
			
			print(f"[Step {self.total_timesteps}] VLM Reward: {mean_vlm_reward:.2f}, "
				  f"VLM Success: {vlm_success_rate*100:.1f}%, Sim Success: {sim_success_rate*100:.1f}%")
			
			# Reset buffers
			self.episode_vlm_rewards = []
			self.episode_sim_rewards = []
			self.episode_sim_success = []
			
			# Log to CSV
			self._log_to_csv(csv_data)
		
		# Evaluation
		if self.n_calls % self.eval_freq == 0:
			eval_data = self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		
		return True
	
	def evaluate(self, agent, deterministic=False):
		"""
		Evaluate agent using SIMULATOR rewards (ground-truth).

		This is the TRUSTWORTHY metric - it measures whether the agent
		actually completes the task according to the simulator, not the VLM.
		"""
		if self.eval_episodes <= 0 or self.eval_env is None:
			return {}

		env = self.eval_env
		eval_vlm_rewards = []
		eval_sim_success = []
		eval_episode_rewards = []

		with torch.no_grad():
			for i in range(self.eval_episodes):
				obs = env.reset()
				episode_done = np.zeros(obs.shape[0], dtype=bool)
				episode_rewards = np.zeros(obs.shape[0])

				for step_idx in range(self.max_steps):
					if self.algorithm == 'dsrl_sac':
						action, _ = agent.predict(obs, deterministic=deterministic)
					elif self.algorithm == 'dsrl_na':
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

					if np.all(episode_done):
						break

				# If any envs didn't finish within max_steps, count them as failures
				for j in range(obs.shape[0]):
					if not episode_done[j]:
						eval_sim_success.append(0)
						eval_episode_rewards.append(episode_rewards[j])

				print(f'Eval episode {i} at timestep {self.total_timesteps}')

		# Compute eval metrics — always log, even if VLM metrics are empty
		eval_data = {
			'timestep': self.total_timesteps,
			'eval_sim_success_rate': np.mean(eval_sim_success) if eval_sim_success else 0,
			'eval_episode_reward': np.mean(eval_episode_rewards) if eval_episode_rewards else 0,
			'eval_vlm_reward': np.mean(eval_vlm_rewards) if eval_vlm_rewards else 0,
			'eval_vlm_success_rate': np.mean([
				1 if s >= self.vlm_success_threshold else 0 for s in eval_vlm_rewards
			]) if eval_vlm_rewards else 0,
		}

		if self.use_wandb:
			name = 'eval_det' if deterministic else 'eval'
			wandb.log({
				f"{name}/sim_success_rate": eval_data['eval_sim_success_rate'],
				f"{name}/episode_reward": eval_data['eval_episode_reward'],
				f"{name}/vlm_reward": eval_data['eval_vlm_reward'],
				f"{name}/vlm_success_rate": eval_data['eval_vlm_success_rate'],
				"timestep": self.total_timesteps,
			}, step=self.total_timesteps)

		print(f"[Eval @ step {self.total_timesteps}] "
			  f"Sim Success: {eval_data['eval_sim_success_rate']*100:.1f}% | "
			  f"Ep Reward: {eval_data['eval_episode_reward']:.2f}" +
			  (f" | VLM Reward: {eval_data['eval_vlm_reward']:.2f}" if eval_vlm_rewards else ""))

		# Log to CSV
		self._log_to_csv(eval_data)

		return eval_data
	
	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps


def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()