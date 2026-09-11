#!/usr/bin/env python3
"""
FPS Sensitivity Analysis for RoboReward Model - Mixed Task Version
===================================================================

This script handles mixed-task datasets where each video has its own task instruction.
It reads task metadata from samples.json files and uses the correct task instruction
for each video during inference.

Unlike the standard version which assumes all videos share the same task, this version:
- Reads samples.json to get per-video task descriptions
- Uses the correct task instruction for each video
- Handles datasets with multiple different tasks (e.g., mixed DROID data)

Author: Platon Karageorgis
"""

import os
import sys
import glob
import json
import re
import argparse
import cv2
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================================================================
# CONFIGURATION DATACLASSES (Abstract for future datasets)
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset to evaluate."""
    name: str
    success_path: str
    failure_path: str
    task_instruction: str
    video_pattern: str = "*.mp4"  # glob pattern for video files
    recursive_search: bool = True

    def __post_init__(self):
        """Validate paths exist."""
        for path, label in [(self.success_path, "success"), (self.failure_path, "failure")]:
            if not os.path.exists(path):
                print(f"⚠️  Warning: {label} path does not exist: {path}")


@dataclass
class RoboRewardDatasetConfig(DatasetConfig):
    """Extended configuration for RoboReward datasets."""
    task_slug: str = ""
    matching_info_path: Optional[str] = None

    @classmethod
    def from_task_download(
        cls,
        task_slug: str,
        base_path: str = "/var/scratch/pkarageo/roboreward_tasks"
    ):
        """
        Auto-configure from downloaded task directory.

        Args:
            task_slug: Slug name of the task (e.g., "pot_on_plate_task")
            base_path: Base directory containing task downloads

        Returns:
            RoboRewardDatasetConfig instance
        """
        task_dir = os.path.join(base_path, task_slug)

        # Validate task directory exists
        if not os.path.exists(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        # Read metadata
        metadata_path = os.path.join(task_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Build config
        return cls(
            name=f"RoboReward_{task_slug}",
            success_path=os.path.join(task_dir, 'success'),
            failure_path=os.path.join(task_dir, 'failure'),
            task_instruction=metadata['task_instruction'],
            video_pattern="*.mp4",  # Flat structure (no nested dirs)
            recursive_search=False,  # Videos directly in success/failure
            task_slug=task_slug,
            matching_info_path=os.path.join(task_dir, 'matching_info.json')
        )


@dataclass  
class ExperimentConfig:
    """Configuration for the FPS experiment."""
    model_cache_path: str
    fps_values: List[float]
    output_dir: str
    include_default_fps: bool = True  # Include the library's default (no fps specified)
    score_threshold: float = 3.5  # Scores >= threshold = success, < threshold = failure
    save_detailed_results: bool = True
    
    # Derived from qwen_vl_utils source code
    DEFAULT_FPS: float = 2.0  # The hardcoded default in qwen_vl_utils


@dataclass
class PredictionResult:
    """Single prediction result."""
    video_path: str
    actual_label: str  # "success" or "failure"
    predicted_label: str
    raw_score: float
    fps_used: float
    num_frames: int
    video_duration: float


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""
    tp: int = 0  # True Positive: actual=success, predicted=success
    fn: int = 0  # False Negative: actual=success, predicted=failure
    fp: int = 0  # False Positive: actual=failure, predicted=success
    tn: int = 0  # True Negative: actual=failure, predicted=failure
    
    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0
    
    def update(self, actual: str, predicted: str):
        """Update confusion matrix with a single prediction."""
        if actual == "success" and predicted == "success":
            self.tp += 1
        elif actual == "success" and predicted == "failure":
            self.fn += 1
        elif actual == "failure" and predicted == "success":
            self.fp += 1
        else:  # actual == "failure" and predicted == "failure"
            self.tn += 1
    
    def to_dict(self) -> Dict:
        return {
            "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn,
            "accuracy": self.accuracy, "precision": self.precision,
            "recall": self.recall, "f1_score": self.f1_score,
            "specificity": self.specificity
        }


@dataclass
class FPSExperimentResult:
    """Results for a single FPS setting."""
    fps: float
    fps_label: str  # "default (2.0)" or "manual: X"
    confusion_matrix: ConfusionMatrix
    predictions: List[PredictionResult] = field(default_factory=list)
    
    def add_prediction(self, result: PredictionResult):
        self.predictions.append(result)
        self.confusion_matrix.update(result.actual_label, result.predicted_label)


# ============================================================================
# MODEL HANDLER
# ============================================================================

class RoboRewardModel:
    """Wrapper for RoboReward model inference."""
    
    REWARD_PROMPT = """You are a reward model. Score the robot's task completion in the video.
Task: {task_instruction}

Scoring criteria:
1 = Complete failure / No progress
2 = Minimal progress, mostly unsuccessful  
3 = Partial completion / Some success
4 = Mostly successful / Minor issues
5 = Perfect task completion

Respond with ONLY a single number (1-5)."""

    def __init__(self, model_cache_path: str):
        self.model = None
        self.processor = None
        self.model_path = None
        self._load_model(model_cache_path)
    
    def _load_model(self, cache_path: str):
        """Load the RoboReward model from cache."""
        print("⬇️  Loading RoboReward Model...")
        try:
            snapshot_subfolders = glob.glob(os.path.join(cache_path, "*"))
            snapshot_subfolders.sort()
            self.model_path = snapshot_subfolders[0]
            
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map="auto",
                trust_remote_code=False
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            print(f"✅ Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
            sys.exit(1)
    
    def predict(
        self, 
        video_path: str, 
        task_instruction: str, 
        fps: Optional[float] = None
    ) -> Tuple[float, int]:
        """
        Run inference on a video.
        
        Args:
            video_path: Path to video file
            task_instruction: Task description
            fps: FPS to use for frame extraction. None = use library default (2.0)
        
        Returns:
            Tuple of (score, num_frames_used)
        """
        # Build video config - fps=None means use library default
        video_config = {"type": "video", "video": video_path}
        if fps is not None:
            video_config["fps"] = fps
        
        messages = [{
            "role": "user",
            "content": [
                video_config,
                {"type": "text", "text": self.REWARD_PROMPT.format(task_instruction=task_instruction)}
            ]
        }]
        
        # Process video
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        
        num_frames = video_inputs[0].shape[0] if video_inputs else 0
        
        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        # Note: Do NOT merge video_kwargs into inputs - it contains fps/do_sample_frames 
        # which are not valid generate() arguments
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=16)
        
        # Decode
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract score
        score = self._extract_score(response)
        return score, num_frames
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from model response."""
        match = re.search(r'([1-5](?:\.\d+)?)', response.strip())
        if match:
            return float(match.group(1))
        # Fallback: try to find any number
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            return min(max(float(numbers[0]), 1.0), 5.0)
        return 3.0  # Default middle score if parsing fails


# ============================================================================
# VIDEO UTILITIES
# ============================================================================

def get_video_info(video_path: str) -> Tuple[float, int, float]:
    """Get video metadata: (fps, total_frames, duration_seconds)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return fps, total_frames, duration


def find_videos(base_path: str, pattern: str = "*.mp4", recursive: bool = True) -> List[str]:
    """Find all video files matching pattern."""
    if recursive:
        search_pattern = os.path.join(base_path, "**", pattern)
        return glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(base_path, pattern)
        return glob.glob(search_pattern)


def load_video_task_mappings(success_path: str, failure_path: str) -> Dict[str, str]:
    """
    Load per-video task descriptions from samples.json files.

    Args:
        success_path: Path to success videos directory
        failure_path: Path to failure videos directory

    Returns:
        Dictionary mapping video filename to task description
    """
    video_to_task = {}

    # Load success samples
    success_samples_path = os.path.join(success_path, 'samples.json')
    if os.path.exists(success_samples_path):
        with open(success_samples_path, 'r') as f:
            success_samples = json.load(f)
        for sample in success_samples:
            video_to_task[sample['video_filename']] = sample['task']
    else:
        print(f"⚠️  Warning: samples.json not found in {success_path}")

    # Load failure samples
    failure_samples_path = os.path.join(failure_path, 'samples.json')
    if os.path.exists(failure_samples_path):
        with open(failure_samples_path, 'r') as f:
            failure_samples = json.load(f)
        for sample in failure_samples:
            video_to_task[sample['video_filename']] = sample['task']
    else:
        print(f"⚠️  Warning: samples.json not found in {failure_path}")

    return video_to_task


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class FPSExperiment:
    """Main experiment runner."""
    
    def __init__(self, config: ExperimentConfig, dataset: DatasetConfig):
        self.config = config
        self.dataset = dataset
        self.model = RoboRewardModel(config.model_cache_path)
        self.results: Dict[str, FPSExperimentResult] = {}

        # Load per-video task mappings from samples.json
        print("\n📋 Loading per-video task descriptions...")
        self.video_to_task = load_video_task_mappings(
            dataset.success_path,
            dataset.failure_path
        )
        print(f"✅ Loaded task descriptions for {len(self.video_to_task)} videos")

        # Prepare output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _get_fps_settings(self) -> List[Tuple[Optional[float], str]]:
        """Get list of (fps_value, label) tuples to test."""
        settings = []
        
        if self.config.include_default_fps:
            settings.append((None, f"default ({self.config.DEFAULT_FPS})"))
        
        for fps in self.config.fps_values:
            settings.append((fps, f"{fps}"))
        
        return settings
    
    def _classify_score(self, score: float) -> str:
        """Convert raw score to binary label."""
        return "success" if score >= self.config.score_threshold else "failure"
    
    def run(self) -> Dict[str, FPSExperimentResult]:
        """Run the full experiment."""
        print("\n" + "="*80)
        print(f"🔬 FPS SENSITIVITY EXPERIMENT (MIXED TASKS)")
        print(f"   Dataset: {self.dataset.name}")
        print(f"   Tasks: {len(set(self.video_to_task.values()))} unique task descriptions")
        print(f"   Note: Using per-video task instructions from samples.json")
        print("="*80)
        
        # Find all videos
        success_videos = find_videos(
            self.dataset.success_path, 
            self.dataset.video_pattern,
            self.dataset.recursive_search
        )
        failure_videos = find_videos(
            self.dataset.failure_path,
            self.dataset.video_pattern, 
            self.dataset.recursive_search
        )
        
        print(f"\n📁 Found {len(success_videos)} success videos")
        print(f"📁 Found {len(failure_videos)} failure videos")
        
        if not success_videos and not failure_videos:
            print("❌ No videos found! Check paths.")
            return {}
        
        # Run for each FPS setting
        fps_settings = self._get_fps_settings()
        
        for fps_value, fps_label in fps_settings:
            print(f"\n{'='*60}")
            print(f"🎬 Testing FPS: {fps_label}")
            print("="*60)
            
            result = FPSExperimentResult(
                fps=fps_value if fps_value else self.config.DEFAULT_FPS,
                fps_label=fps_label,
                confusion_matrix=ConfusionMatrix()
            )
            
            # Process success videos
            print(f"\n  Processing SUCCESS videos ({len(success_videos)})...")
            for i, video_path in enumerate(success_videos, 1):
                self._process_video(video_path, "success", fps_value, result, i, len(success_videos))
            
            # Process failure videos
            print(f"\n  Processing FAILURE videos ({len(failure_videos)})...")
            for i, video_path in enumerate(failure_videos, 1):
                self._process_video(video_path, "failure", fps_value, result, i, len(failure_videos))
            
            self.results[fps_label] = result
            self._print_fps_summary(result)
        
        return self.results
    
    def _process_video(
        self,
        video_path: str,
        actual_label: str,
        fps: Optional[float],
        result: FPSExperimentResult,
        idx: int,
        total: int
    ):
        """Process a single video and update results."""
        try:
            # Get the correct task instruction for this specific video
            video_filename = os.path.basename(video_path)
            task_instruction = self.video_to_task.get(
                video_filename,
                self.dataset.task_instruction  # Fallback to generic if not found
            )

            video_fps, total_frames, duration = get_video_info(video_path)
            score, num_frames = self.model.predict(
                video_path,
                task_instruction,  # Use per-video task instruction
                fps
            )
            predicted_label = self._classify_score(score)
            
            prediction = PredictionResult(
                video_path=video_path,
                actual_label=actual_label,
                predicted_label=predicted_label,
                raw_score=score,
                fps_used=fps if fps else self.config.DEFAULT_FPS,
                num_frames=num_frames,
                video_duration=duration
            )
            result.add_prediction(prediction)
            
            # Status indicator
            status = "✓" if actual_label == predicted_label else "✗"
            print(f"    [{idx}/{total}] {status} Score: {score:.1f} → {predicted_label} (actual: {actual_label})")
            
        except Exception as e:
            print(f"    [{idx}/{total}] ❌ Error: {e}")
    
    def _print_fps_summary(self, result: FPSExperimentResult):
        """Print summary for a single FPS setting."""
        cm = result.confusion_matrix
        
        print(f"\n  📊 CONFUSION MATRIX (FPS: {result.fps_label})")
        print("  " + "-"*40)
        print(f"                    Predicted")
        print(f"                  Success  Failure")
        print(f"  Actual Success    {cm.tp:3d}      {cm.fn:3d}")
        print(f"  Actual Failure    {cm.fp:3d}      {cm.tn:3d}")
        print("  " + "-"*40)
        print(f"  Accuracy:    {cm.accuracy:.2%}")
        print(f"  Precision:   {cm.precision:.2%}")
        print(f"  Recall:      {cm.recall:.2%}")
        print(f"  F1 Score:    {cm.f1_score:.2%}")
        print(f"  Specificity: {cm.specificity:.2%}")
    
    def print_comparison_table(self):
        """Print a comparison table across all FPS settings."""
        print("\n" + "="*80)
        print("📈 COMPARISON ACROSS ALL FPS SETTINGS")
        print("="*80)
        
        # Header
        print(f"\n{'FPS':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FN':>5} {'FP':>5} {'TN':>5}")
        print("-"*95)
        
        for fps_label, result in self.results.items():
            cm = result.confusion_matrix
            print(f"{fps_label:<20} {cm.accuracy:>10.2%} {cm.precision:>10.2%} {cm.recall:>10.2%} {cm.f1_score:>10.2%} {cm.tp:>5} {cm.fn:>5} {cm.fp:>5} {cm.tn:>5}")
        
        # Find best FPS
        if self.results:
            best_fps = max(self.results.items(), key=lambda x: x[1].confusion_matrix.f1_score)
            print("-"*95)
            print(f"\n🏆 Best FPS setting: {best_fps[0]} (F1: {best_fps[1].confusion_matrix.f1_score:.2%})")
    
    def save_results(self, filename: Optional[str] = None):
        """Save detailed results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fps_experiment_MIXED_{self.dataset.name}_{timestamp}.json"

        output_path = os.path.join(self.config.output_dir, filename)

        # Get unique tasks
        unique_tasks = list(set(self.video_to_task.values()))

        data = {
            "experiment_info": {
                "dataset": self.dataset.name,
                "experiment_type": "mixed_tasks",
                "num_unique_tasks": len(unique_tasks),
                "task_instruction": self.dataset.task_instruction,
                "score_threshold": self.config.score_threshold,
                "timestamp": datetime.now().isoformat(),
                "default_fps_note": f"qwen_vl_utils defaults to {self.config.DEFAULT_FPS} FPS when not specified",
                "note": "Each video evaluated with its specific task instruction from samples.json"
            },
            "results": {}
        }
        
        for fps_label, result in self.results.items():
            data["results"][fps_label] = {
                "fps_value": result.fps,
                "confusion_matrix": result.confusion_matrix.to_dict(),
                "predictions": [
                    {
                        "video": p.video_path,
                        "actual": p.actual_label,
                        "predicted": p.predicted_label,
                        "score": p.raw_score,
                        "frames_used": p.num_frames,
                        "video_duration": p.video_duration
                    }
                    for p in result.predictions
                ] if self.config.save_detailed_results else []
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FPS Confusion Matrix Experiment")
    parser.add_argument(
        '--task-slug',
        type=str,
        default=None,
        help='RoboReward task slug (e.g., "pot_on_plate_task"). If not provided, runs DROID dataset.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (defaults based on dataset type)'
    )
    parser.add_argument(
        '--fps-values',
        nargs='+',
        type=float,
        default=[2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0],
        help='FPS values to test'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================

    if args.task_slug:
        # RoboReward dataset from downloaded task
        print(f"\n🤖 Running experiment on RoboReward task: {args.task_slug}\n")
        dataset_config = RoboRewardDatasetConfig.from_task_download(args.task_slug)
        default_output_dir = "/var/scratch/pkarageo/roboreward_results"
    else:
        # Default DROID dataset
        print(f"\n🤖 Running experiment on DROID dataset\n")
        dataset_config = DatasetConfig(
            name="DROID_pillows",
            success_path="/var/scratch/pkarageo/my_success_case",
            failure_path="/var/scratch/pkarageo/my_failure_case",
            task_instruction="rearrange pillows on sofa",
            video_pattern="22246076.mp4",  # Specific camera view
            recursive_search=True
        )
        default_output_dir = "/var/scratch/pkarageo/roboreward_results"

    # Use provided output dir or default
    output_dir = args.output_dir if args.output_dir else default_output_dir

    # ========================================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================================

    experiment_config = ExperimentConfig(
        model_cache_path="/var/scratch/pkarageo/hf_cache/models--teetone--RoboReward-8B/snapshots",
        fps_values=args.fps_values,
        output_dir=output_dir,
        include_default_fps=True,  # Also test with fps=None (library default)
        score_threshold=3.5,  # >=3.5 = success, <3.5 = failure
        save_detailed_results=True
    )

    # ========================================================================
    # RUN EXPERIMENT
    # ========================================================================

    experiment = FPSExperiment(experiment_config, dataset_config)
    results = experiment.run()

    if results:
        experiment.print_comparison_table()
        experiment.save_results()

    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
