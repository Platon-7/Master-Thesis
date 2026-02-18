"""
RoboReward VLM Wrapper for DSRL

This module provides a VLM-based reward function using RoboReward-8B.
It scores robot episodes based on task progress (1-5 discrete scale).

Usage:
    - Initialize once at the start of training
    - Call `compute_reward(images, task_instruction)` at the end of each episode
"""

import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union


class RoboRewardModel:
    """
    Wrapper for RoboReward-8B VLM that scores robot task execution.
    
    Scores:
        1 - No Success: Final state shows no goal-relevant change.
        2 - Minimal Progress: Final state shows small but insufficient change.
        3 - Partial Completion: Good progress but violates a major requirement.
        4 - Near Completion: Correct but misses a minor requirement.
        5 - Perfect Completion: Satisfies all requirements.
    """
    
    CRITIC_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change.
2 - Minimal Progress: Final state shows small but insufficient change.
3 - Partial Completion: Good progress but violates a major requirement.
4 - Near Completion: Correct but misses a minor requirement.
5 - Perfect Completion: Satisfies all requirements.
"""

    # Task instructions for Robomimic environments
    TASK_INSTRUCTIONS = {
        "lift": "Pick up the red cube from the table",
        "can": "Pick up the can and place it in the correct bin",
        "square": "Pick up the square nut and place it on the peg",
        "transport": "Transport the payload to the target location",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",  # Default to single GPU; override via config for multi-GPU
        dtype: torch.dtype = torch.float16,
        max_frames: int = 32,  # Max frames to send to VLM (subsample if needed)
    ):
        """
        Initialize RoboReward model.
        
        Args:
            model_path: Path to model or HF cache. If None, auto-detect from cache.
            device: Device to run on. Use "cuda:1" to separate from DSRL on cuda:0 in multi-GPU setups
            dtype: Model dtype (float16 recommended for memory)
            max_frames: Maximum frames to send to VLM per episode
        """
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info
        
        self.device = device
        self.dtype = dtype
        self.max_frames = max_frames
        self.process_vision_info = process_vision_info
        
        # Auto-detect model path from HF cache
        if model_path is None:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            base_cache = os.path.join(hf_home, "models--teetone--RoboReward-8B", "snapshots")
            snapshot_subfolders = glob.glob(os.path.join(base_cache, "*"))
            snapshot_subfolders.sort()
            model_path = snapshot_subfolders[0] if snapshot_subfolders else None
            
        if model_path is None:
            raise ValueError("Could not find RoboReward model. Please specify model_path.")
        
        print(f"Loading RoboReward-8B from {model_path}...")
        print(f"   Target device: {device}")
        print(f"   Using 4-bit quantization (NF4)")
        
        # 4-bit quantization config (same as your local_server.py)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Determine device_map based on requested device
        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            device_map = {"": gpu_id}  # Put entire model on specified GPU
        else:
            device_map = "auto"
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=False,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        
        print(f"RoboReward-8B loaded on {device}")
        
    def _subsample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Subsample frames if there are too many."""
        if len(frames) <= self.max_frames:
            return frames
        
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def compute_reward(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        task_instruction: str,
        return_raw_score: bool = False,
    ) -> float:
        """
        Compute VLM reward for an episode.
        
        Args:
            frames: List of PIL Images or numpy arrays (RGB) from the episode
            task_instruction: Natural language description of the task
            return_raw_score: If True, return raw 1-5 score. If False, return normalized reward.
            
        Returns:
            Reward value. If return_raw_score=False, returns normalized to [-1, 0] range
            where -1 = score 1 (failure), 0 = score 5 (success)
        """
        # Convert numpy arrays to PIL Images if needed
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)
        
        # Subsample if too many frames
        pil_frames = self._subsample_frames(pil_frames)
        
        # Prepare VLM input
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.CRITIC_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": f"Task: {task_instruction}"},
            ]}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages, return_video_kwargs=True
        )
        
        # Process video inputs - video_inputs is a list of videos (each a list of PIL frames)
        videos = video_inputs if video_inputs else None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)
        
        # Generate score (deterministic)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Deterministic
            )
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        
        # Parse score from output
        score_match = re.search(r'ANSWER:\s*([1-5])', output_text)
        if not score_match:
            score_match = re.search(r'\b([1-5])\b', output_text)
        
        raw_score = int(score_match.group(1)) if score_match else 1  # Default to 1 if parse fails
        
        # Log first few VLM calls for verification (then reduce frequency)
        if not hasattr(self, '_vlm_call_count'):
            self._vlm_call_count = 0
        self._vlm_call_count += 1
        if self._vlm_call_count <= 5 or self._vlm_call_count % 50 == 0:
            print(f"[VLM #{self._vlm_call_count}] Task: {task_instruction}")
            print(f"    Frames: {len(frames)}, Raw output: {output_text[-100:]}")  # Last 100 chars
            print(f"    Parsed discrete score: {raw_score}/5 → Reward: {(raw_score - 5) / 4.0:.2f}")
        
        if return_raw_score:
            return float(raw_score)
        
        # Normalize to [-1, 0] range (compatible with DSRL's sparse reward setup)
        # Score 1 -> -1, Score 5 -> 0
        normalized_reward = (raw_score - 5) / 4.0  # Maps [1,5] to [-1, 0]
        
        return normalized_reward
    
    def get_task_instruction(self, env_name: str) -> str:
        """Get the task instruction for a Robomimic environment."""
        return self.TASK_INSTRUCTIONS.get(env_name, f"Complete the {env_name} task")


# Singleton instance for efficiency (avoid reloading model)
_roboreward_instance: Optional[RoboRewardModel] = None


def get_roboreward_model(**kwargs) -> RoboRewardModel:
    """Get or create the singleton RoboReward model instance."""
    global _roboreward_instance
    if _roboreward_instance is None:
        _roboreward_instance = RoboRewardModel(**kwargs)
    return _roboreward_instance


if __name__ == "__main__":
    # Quick test
    print("Testing RoboReward wrapper...")
    
    # Create dummy frames (just for testing the interface)
    dummy_frames = [Image.new('RGB', (224, 224), color='red') for _ in range(10)]
    
    model = get_roboreward_model()
    reward = model.compute_reward(dummy_frames, "Pick up the red cube")
    print(f"Test reward: {reward}")
