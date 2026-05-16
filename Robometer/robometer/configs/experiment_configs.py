#!/usr/bin/env python3
"""
Experiment configurations for RBM training.
This file contains all the dataclass configurations that can be reused
across different training scripts.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# ModelConfig used to inherit from transformers.PretrainedConfig, but OmegaConf's
# structured-config system can't handle PretrainedConfig's Union[str, torch.dtype]
# type hints (raises "Unions of containers are not supported"). Grepping the codebase
# shows no actual use of PretrainedConfig's methods on ModelConfig — the inheritance was
# vestigial — so we drop it.


@dataclass
class ModelConfig:
    """Config for model settings"""

    base_model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_type: str = field(default="default")
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)
    train_vision_encoder: bool = field(default=False, metadata={"help": "Whether to train the vision encoder"})
    train_language_model: bool = field(default=True, metadata={"help": "Whether to train the language model"})

    train_progress_head: bool = field(default=False, metadata={"help": "Whether to train the progress prediction head"})
    train_preference_head: bool = field(
        default=False, metadata={"help": "Whether to train the preference prediction head"}
    )
    train_success_head: bool = field(default=False, metadata={"help": "Whether to train the success prediction head"})

    average_temporal_patches: bool = field(
        default=False,
        metadata={
            "help": "If True, average all tokens within each temporal patch group for progress prediction. If False, use the last token (boundary) of each temporal patch group."
        },
    )

    frame_pooling: str = field(
        default="mean",
        metadata={
            "help": "How to pool vision tokens into a per-frame embedding for progress/success heads in multi-image mode. "
            "Options: 'mean' (average over patch tokens), 'boundary' (use last patch token), 'attention' (learned attention pooling)."
        },
    )
    frame_pooling_attn_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for attention pooling over patch tokens (only used when frame_pooling='attention')."
        },
    )

    use_per_frame_progress_token: bool = field(
        default=False,
        metadata={
            "help": "If True, add a <|prog_token|> after each frame and use those token embeddings "
            "for per-frame progress prediction. Requires use_multi_image=True."
        },
    )

    progress_head_mode: str = field(
        default="c51",
        metadata={
            "help": "Shape of the per-frame progress head. 'c51' (default) emits N=progress_discrete_bins "
            "logits matching the pretrained Robometer-4B head. 'corn' emits 4 cumulative-threshold "
            "logits per frame for losses.md §Loss 1 (Chris' asymmetric CORN). 'corn' implies a fresh "
            "random-init head since its output dim differs from the checkpoint."
        },
    )

    use_multi_image: bool = field(
        default=False,
        metadata={
            "help": "If True, feed frames as a list of images instead of converting to video (avoids encoding overhead). "
            "If False, detect automatically based on number of vision_start tokens in the sequence."
        },
    )

    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT/LoRA or train full model"})
    peft_vision_encoder: bool = field(default=False, metadata={"help": "Whether to attach LoRA to the vision encoder"})

    # use bitsandbytes for quantization
    quantization: bool = field(default=False, metadata={"help": "Whether to use bitsandbytes for quantization"})

    # use unsloth for faster training
    use_unsloth: bool = field(
        default=False, metadata={"help": "Whether to use unsloth for faster vision model training"}
    )

    progress_loss_type: str = field(
        default="l2",
        metadata={"help": "Type of progress loss: 'l1', 'l2', or 'discrete'"},
    )
    progress_discrete_bins: Optional[int] = field(
        default=None,
        metadata={"help": "Number of discrete bins for progress when using discrete loss (None for continuous)"},
    )
    # CORN-head bias-init priors. Mirrored on LossConfig (canonical home) but lives here too
    # because heads.py constructs the head with only ModelConfig in scope. train.py copies the
    # value from cfg.loss.corn_bias_init_priors before model construction. See LossConfig.
    corn_bias_init_priors: Optional[List[float]] = field(default=None)
    # rewind sub-config
    rewind: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        from robometer.models.rewind_transformer import ReWINDTransformerConfig

        if self.rewind is not None and isinstance(self.rewind, dict):
            # Pass progress_loss_type and progress_discrete_bins from parent config if not set in rewind dict
            if "progress_loss_type" not in self.rewind:
                self.rewind["progress_loss_type"] = self.progress_loss_type
            if "progress_discrete_bins" not in self.rewind:
                self.rewind["progress_discrete_bins"] = self.progress_discrete_bins or 10
            if "use_per_frame_progress_token" not in self.rewind:
                self.rewind["use_per_frame_progress_token"] = self.use_per_frame_progress_token

            self.rewind = ReWINDTransformerConfig(**self.rewind)


@dataclass
class PEFTConfig:
    """Config for PEFT/LoRA settings"""

    r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default="none")
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    peft_vision_encoder: bool = field(default=False, metadata={"help": "Whether to attach LoRA to the vision encoder"})


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    # Dataset paths
    train_datasets: List[str] = field(
        default_factory=lambda: ["abraranwar/libero_rfm"], metadata={"help": "List of training dataset names"}
    )
    eval_datasets: List[str] = field(
        default_factory=lambda: ["abraranwar/libero_rfm"], metadata={"help": "List of evaluation dataset names"}
    )

    # Dataset type and configuration
    dataset_type: str = field(
        default="default",
        metadata={"help": "Dataset type: 'default', 'rewound', 'success_failure'"},
    )

    max_frames_after_preprocessing: int = field(
        default=64, metadata={"help": "Maximum number of frames to extract from videos after preprocessing"}
    )
    max_frames: int = field(default=8, metadata={"help": "Maximum number of frames to extract from videos"})
    min_frames_per_trajectory: int = field(
        default=5,
        metadata={
            "help": "Minimum number of frames required per trajectory (trajectories with fewer frames will be filtered out)"
        },
    )
    resized_height: Optional[int] = field(default=None, metadata={"help": "Height to resize video frames to"})
    resized_width: Optional[int] = field(default=None, metadata={"help": "Width to resize video frames to"})

    # Video/image processing mode
    use_multi_image: bool = field(
        default=False,
        metadata={
            "help": "If True, feed frames as a list of images instead of converting to video. "
            "This avoids video encoding overhead and works for both SmolVLM and Qwen models."
        },
    )
    stratified_batch_balance: bool = field(
        default=False,
        metadata={
            "help": "If True, every training batch is constructed as exact 50/50 failure↔success "
            "via the StratifiedWarmupBatchSampler. Combined with warmup_steps for two-phase training."
        },
    )
    warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of initial training steps to draw batches exclusively from failure samples "
            "whose data_source contains 'warmup' (falls back to any failure if no warmup pool is registered). "
            "After this many steps the stratified 50/50 sampling kicks in. Ignored unless "
            "stratified_batch_balance=True."
        },
    )
    use_icl: bool = field(
        default=False,
        metadata={
            "help": "If True, each sample can carry a 16-frame successful 'demo' trajectory prepended to the query as "
            "an in-context example. Requires pairs_index_path."
        },
    )
    icl_prob: float = field(
        default=0.0,
        metadata={
            "help": "Per-sample probability (0..1) of populating the demo when use_icl=True. "
            "1.0 = always-on, 0.5 = coin flip, 0.0 = never."
        },
    )
    pairs_index_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a JSONL where each row maps a training episode_id to its partner (task, frames_path::episode_dir). "
            "Required when use_icl=True."
        },
    )
    icl_min_coverage: float = field(
        default=0.80,
        metadata={
            "help": "Minimum partner-coverage rate per training pool when use_icl=True. Coverage is "
            "(# episodes with a non-null partner in pairs_index_path) / (# episodes in the HF dataset). "
            "This is NOT the ICL-on rate — that is coverage × data.icl_prob (e.g. 96% coverage × 0.5 "
            "icl_prob → ~48% effective ICL-on). The threshold is a drift sniff: catches catastrophic "
            "mismatch between the HF Arrow dataset and the pair index. The Apr-25 bake-off saw 53% "
            "coverage where ~100% was expected (HF built from an older snapshot of train.jsonl) — "
            "0.80 catches that unambiguously. After the Apr-30 success-pairing pass every split sits "
            "at 92-100% coverage, so 0.80 leaves a comfortable margin. Set lower for legitimate "
            "low-coverage scenarios (Phase 2 broad data sweeps); set to 0.0 to disable entirely."
        },
    )
    failure_label_smoothing: str = field(
        default="none",
        metadata={
            "help": "Post-process curated failure frame_labels before use as target_progress. "
            "'none' = rubric decimals as-is (stair-step). 'linear' = piecewise-linear ramp "
            "between label transitions to match the smooth t/T character of success labels."
        },
    )
    icl_demo_blank: bool = field(
        default=False,
        metadata={
            "help": "Ablation: when True, replace demo frames with all-zero pixels after loading. "
            "Keeps demo length, separator token, and position embeddings intact, but removes the "
            "visual content. Used to test whether the VLM is genuinely using the demo's visual "
            "signal vs. just reacting to longer context length / extra separator token. "
            "Default False."
        },
    )
    icl_task_dropout: bool = field(
        default=False,
        metadata={
            "help": "When True AND the sample has an ICL demo attached, stochastically corrupt "
            "the natural-language task instruction in the prompt (per Chris's recipe). With p=1/3 "
            "keep original; p=1/3 replace with a generic string (\"\", \"do the task\", or "
            "\"perform the demonstrated task\"); p=1/3 split by whitespace and drop each word "
            "independently with prob 0.1. Goal: encourage the VLM to rely on the demo trajectory "
            "rather than memorizing the exact task wording. Only applies during training (not eval) "
            "and only when context_trajectory is populated. No-op when use_icl=False. "
            "If icl_task_dropout_sources is non-empty, this global flag is IGNORED and the "
            "per-source list controls activation instead."
        },
    )
    icl_task_dropout_sources: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Per-data_source restriction for icl_task_dropout. When non-empty, the "
            "corruption recipe (1/3 keep, 1/3 generic, 1/3 word-drop) is applied ONLY to samples "
            "whose data_source is in this list — every other source keeps its original task text. "
            "Use case: droid uses a strong text shortcut (95% of failure tasks also appear as "
            "successes — see analysis); we want to ablate this for droid specifically without "
            "disrupting metaworld/robometer/failsafe (which DO use vision discriminatively). "
            "Set e.g. [droid] to target one source. Empty (default) → falls back to the global "
            "icl_task_dropout flag's behavior."
        },
    )
    success_label_noise_std: float = field(
        default=0.0,
        metadata={
            "help": "If > 0, add per-frame N(0, sigma^2) Gaussian noise to success target_progress "
            "as a regularizer against the over-clean monotone t/T signal. Clipped to [0, 1]."
        },
    )
    failure_oversample_factor: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplicative weight on failure samples for a WeightedRandomSampler. "
                "1.0 = natural ratio (no oversampling). >1.0 tilts the draw toward failures. "
                "Active only when data.stratified_batch_balance is False AND value > 1.0; "
                "otherwise ignored. Produces mixed (not pure-class) batches."
            ),
        },
    )
    success_oversample_per_source: Dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Per-data_source multiplier applied to SUCCESS-trajectory weights in the "
                "WeightedRandomSampler. Useful when a particular source (e.g. metaworld, with only "
                "1,946 successes vs 27,010 failures) is severely undersampled at default weights "
                "and the model never sees enough successes to learn its target. The multiplier is "
                "applied on top of the base success weight (1.0) — so a value of 13.0 makes each "
                "metaworld-success ~13× more likely to be drawn than each non-metaworld success of "
                "equal natural-weight. Does NOT change failure weights. "
                "Empty dict {} = no per-source boost (default; legacy behavior). "
                "Activates the WeightedRandomSampler path even when failure_oversample_factor==1.0, "
                "so this knob alone is sufficient to enable mixed-weight sampling. "
                "Default is empty dict (not None) so OmegaConf can merge `++` dict overrides into "
                "it — merging a dict literal into a None-typed structured-config field crashes "
                "with AttributeError in _map_merge."
            ),
        },
    )
    failure_oversample_per_source: Dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Per-data_source multiplier applied to FAILURE-trajectory weights in the "
                "WeightedRandomSampler — symmetric of success_oversample_per_source. Useful for "
                "sources with severe within-source success/failure imbalance in the opposite "
                "direction from metaworld: droid has 155k successes but only 6k failures, so the "
                "model's gradient signal is dominated by successes and it never learns to push "
                "failure predictions DOWN. Setting droid: 9.85 (= 155306/6038/2.61) brings "
                "droid succ:fail to ~1:1 within droid draws (combined with the existing failure "
                "_oversample_factor of 2.61), forcing the model to visually discriminate. "
                "The multiplier is applied on top of the global failure_oversample_factor — so the "
                "effective per-trajectory weight for a droid failure is "
                "failure_oversample_factor × failure_oversample_per_source['droid']. "
                "Empty dict {} = no per-source boost (default; legacy behavior). "
                "Activates the WeightedRandomSampler path even when failure_oversample_factor==1.0."
            ),
        },
    )
    keep_orphan_task_failures: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, bypass _filter_task_based_criteria so trajectories from tasks that have "
                "zero `optimal`/`successful` demos are kept in the training pool. Default False "
                "preserves the upstream recipe assumption that every task has a positive anchor. "
                "Safe to enable when sample_type_ratio=[0,1,0], stratified_batch_balance=False, "
                "failure_kl_weight=0, and preference head is off — the progress sampler only "
                "operates on each trajectory's own frame_labels and any cross-task lookups iterate "
                "optimal_by_task.keys() (so orphan tasks are naturally skipped as candidates)."
            )
        },
    )
    traj_same_source_prob: float = field(
        default=0.5,
        metadata={
            "help": "Probability of sampling a different task instruction from the same data source "
            "when generating negative instructions. Remaining probability samples across all sources."
        },
    )
    shuffle_progress_frames: bool = field(
        default=False,
        metadata={
            "help": "If True, shuffle progress trajectory frames (except the first frame) "
            "and their corresponding target progress labels during training for RBM heads."
        },
    )

    use_per_frame_progress_token: bool = field(
        default=False,
        metadata={
            "help": "If True, add a <|prog_token|> after each frame for per-frame progress prediction. "
            "Requires use_multi_image=True."
        },
    )

    # Data generation parameters
    sample_type_ratio: List[float] = field(
        default_factory=lambda: [1, 1, 1], metadata={"help": "Ratio of pref and progress samples"}
    )
    dataset_preference_ratio: float = field(
        default=0.8, metadata={"help": "Ratio of dataset preference samples to generated preference samples"}
    )
    # [rewind, suboptimal_same_task, different_task, reverse_progress]
    preference_strategy_ratio: List[float] = field(default_factory=lambda: [1, 1, 1, 1])
    # [different_task, forward_progress, reverse_progress, rewind]
    progress_strategy_ratio: List[float] = field(default_factory=lambda: [1, 1, 1, 1])

    data_source_weights: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": "Dictionary mapping data source names to sampling weights (e.g., {'metaworld': 0.2, 'libero': 0.8})"
        },
    )

    # Processing parameters
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the dataset"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    # Evaluation parameters
    eval_subset_size: Optional[int] = field(default=None, metadata={"help": "Number of samples to use for evaluation"})

    # Dataloader parameters
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin memory in dataloader"})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of worker processes for dataloader"})
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers dataset instances alive."
        },
    )

    max_trajectories: int = field(default=-1, metadata={"help": "Maximum number of trajectories to use for dataset"})

    # Embedding loading parameters
    load_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to load precomputed embeddings instead of processing frames (ReWiND only)"},
    )

    progress_pred_type: str = field(
        default="absolute_wrt_total_frames", metadata={"help": "Type of progress prediction: 'absolute' or 'relative'"}
    )

    # Success prediction thresholds
    min_success: float = field(
        default=0.0, metadata={"help": "Minimum progress threshold for success prediction (label=0, failure)"}
    )
    max_success: float = field(
        default=1.0, metadata={"help": "Maximum progress threshold for success prediction (label=1, success)"}
    )
    dataset_success_cutoff_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataset-specific success cutoff file (CSV format: dataset_name,success_percentage)"},
    )

    # Partial success threshold (for datasets with partial_success like RoboArena and RoboReward)
    partial_success_threshold: float = field(
        default=0.2,
        metadata={
            "help": "Minimum difference in partial_success required between chosen and rejected trajectories for preference sampling"
        },
    )

    # Predict last frame partial progress mask
    predict_last_frame_partial_progress: bool = field(
        default=False,
        metadata={
            "help": "If True, create a mask that marks the last frame for partial_success trajectories (< 1.0). "
            "If False, the mask is all 1.0s (no masking)."
        },
    )

    # Progress loss configuration
    progress_loss_type: str = field(
        default="l2",
        metadata={"help": "Type of progress loss: 'l1', 'l2', or 'discrete' (synced from loss config)"},
    )
    progress_discrete_bins: int = field(
        default=10,
        metadata={"help": "Number of discrete bins for progress when using discrete loss (synced from loss config)"},
    )


@dataclass
class CustomEvaluationConfig:
    """Config for custom evaluation settings"""

    eval_types: List[str] = field(default_factory=lambda: ["policy_ranking", "confusion_matrix", "reward_alignment"])
    policy_ranking: List[str] = field(default_factory=lambda: ["aliangdw_metaworld_metaworld_eval"])
    confusion_matrix: List[str] = field(default_factory=lambda: ["aliangdw_metaworld_metaworld_eval"])
    reward_alignment: List[str] = field(default_factory=lambda: ["aliangdw_metaworld_metaworld_eval"])
    quality_preference: List[str] = field(default_factory=lambda: ["aliangdw_metaworld_metaworld_eval"])
    comparisons_per_task: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit number of quality preference comparisons per task. None = use all comparisons. Uniformly samples if limit is set."
        },
    )
    max_comparisons: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit total number of quality preference comparisons across all tasks. None = use all comparisons. Uniformly samples if limit is set."
        },
    )
    num_examples_per_quality_pr: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of trajectories to sample per quality label for policy ranking evaluation. Only tasks with multiple quality labels are used. If None = use all."
        },
    )
    num_partial_successes: Optional[int] = field(
        default=None,
        metadata={
            "help": "For RoboArena datasets: Number of total trajectories to sample using circular sampling across partial_success values. None = use num_examples_per_quality_pr per partial_success group."
        },
    )
    policy_ranking_max_tasks: Optional[int] = field(
        default=100,
        metadata={
            "help": "Maximum number of tasks to use for policy ranking evaluation. None = use all tasks with multiple quality labels."
        },
    )
    policy_ranking_dual_icl: bool = field(
        default=False,
        metadata={
            "help": "If True, every policy_ranking eval round runs twice — once with icl_prob "
            "temporarily forced to 1.0 (the 'icl-on' pass — every paired sample gets a demo) and "
            "once with icl_prob forced to 0.0 (the 'icl-off' pass — no demos). Metric keys gain "
            "`_iclon` / `_icloff` suffixes. Doubles eval cost but lets the trainer track ICL "
            "benefit live across training. Requires use_icl=true and a populated pair index. "
            "NB: the training-time data.icl_prob (typically 0.5 for coin-flipped training) is not "
            "used during eval — we want deterministic comparison, so eval forces 1 vs 0."
        },
    )
    custom_eval_random_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for sampling trajectories in custom evaluation samplers. Ensures all ranks sample the same trajectories to prevent hangs when dataloaders have different lengths."
        },
    )
    reward_alignment_max_trajectories: Optional[int] = field(
        default=10,
        metadata={
            "help": "Maximum number of trajectories to use for reward alignment evaluation. None = use all trajectories."
        },
    )
    use_frame_steps: bool = field(
        default=True,
        metadata={
            "help": "Whether to use frame steps (subsequences) for reward_alignment and policy_ranking evaluations. True = generate subsequences (0:frame_step, 0:2*frame_step, etc.), False = use whole trajectory."
        },
    )
    subsample_n_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Number of frames to subsample for reward alignment evaluation. null = use all frames."},
    )
    pad_frames: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad frames for reward alignment and policy ranking evaluations. True = pad frames, False = use original frames."
        },
    )


@dataclass
class TrainingConfig:
    """Config for training settings"""

    # Hardware settings
    num_gpus: int = field(default=2, metadata={"help": "Number of GPUs to use for training"})

    # Output and logging
    output_dir: str = field(default="./logs")
    exp_name: str = field(default="rbm")
    max_seq_length: int = field(default=1024)
    beta: float = field(default=0.1)
    resume_from_checkpoint: Optional[str] = field(default=None)
    load_from_checkpoint: Optional[str] = field(default=None)
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "If True, overwrite the output directory if it exists. If False, raise an error if it exists."
        },
    )

    # Training arguments
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=5e-7)
    num_train_epochs: Optional[int] = field(default=1)  # Default to 1 epoch if not specified
    save_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    bf16: bool = field(default=False)
    fp16: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_bucket_cap_mb: int = field(default=25)
    max_steps: Optional[int] = field(default=-1)  # -1 means no limit, use num_train_epochs instead
    save_steps: int = field(default=100)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    dataloader_persistent_workers: bool = field(default=False)

    # Evaluation settings
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'"})
    eval_steps: Optional[int] = field(
        default=None, metadata={"help": "Number of steps between evaluations (required if evaluation_strategy='steps')"}
    )
    run_default_eval: bool = field(
        default=False, metadata={"help": "Whether to run default evaluation during training"}
    )
    custom_eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps between custom evaluations (required if evaluation_strategy='steps')"},
    )
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation during training"})
    prediction_loss_only: bool = field(default=True, metadata={"help": "Only compute loss for the prediction head"})

    # Optimizer settings
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=0)
    warmup_ratio: float = field(default=0.1)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for optimizer"})

    # Vision encoder fine-tuning settings
    vision_encoder_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for last N vision encoder layers. If None, uses the same LR as other parameters."
        },
    )
    vision_encoder_num_layers: int = field(
        default=2,
        metadata={
            "help": "Number of last vision encoder layers to fine-tune with vision_encoder_lr. Only used if vision_encoder_lr is set."
        },
    )

    # RBM-specific settings
    predict_pref_progress: bool = field(
        default=False, metadata={"help": "Whether to predict progress for preference samples"}
    )


@dataclass
class LossConfig:
    """Config for loss computation settings"""

    success_positive_weight: float = field(
        default=1.0,
        metadata={"help": "Positive class weight for BCEWithLogits loss in success prediction (pos_weight)."},
    )
    predict_last_frame_progress: bool = field(
        default=False,
        metadata={"help": "If True, only compute progress loss for the last frame in the sequence"},
    )
    progress_loss_type: str = field(
        default="l2",
        metadata={
            "help": "Type of progress loss: 'l1', 'l2', 'discrete', or 'c51_asymmetric' "
            "(Chris' asymmetric C51; see losses.md §Loss 2)."
        },
    )
    progress_discrete_bins: int = field(
        default=10,
        metadata={
            "help": "Number of discrete bins for progress when progress_loss_type is 'discrete' or "
            "'c51_asymmetric' (default: 10; matches the Robometer-4B checkpoint)."
        },
    )
    success_loss_type: str = field(
        default="bce_balanced",
        metadata={
            "help": "Type of success loss: 'bce_balanced' (Robometer's dynamic class-balanced BCE) "
            "or 'bce_asymmetric' (Chris' slide: mean(bce[neg]) + lambda * mean(bce[pos]) with "
            "an empty-positives guard)."
        },
    )
    asymmetric_lambda: float = field(
        default=0.3,
        metadata={
            "help": "Single lambda shared by 'c51_asymmetric' and 'bce_asymmetric'. lambda < 1 "
            "damps the underestimation / positive-class side so the model emits fewer false "
            "positives — matches Chris' slide."
        },
    )
    corn_asymmetric_c: float = field(
        default=1.5,
        metadata={
            "help": "Curvature of the per-threshold asymmetric weight for corn_asymmetric loss. "
            "alpha_k = 1 + c*(k-2) for k in {2,3,4,5}, so alpha_5 = 1 + 3c is largest (penalising "
            "'false terminal-success' hardest). c=0 recovers symmetric CORN."
        },
    )
    corn_pos_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Per-threshold positive-class weights β_k for the CORN loss. List of 4 floats "
            "applied as multipliers on the positive-side BCE term: pos_term = β_k · b · log σ. "
            "Used to combat class imbalance on rare upper thresholds (P(y≥5) ≈ 6%, P(y≥4) ≈ 20% "
            "in our train pool — symmetric BCE collapses the head to constant-near-zero). "
            "Recommended: sqrt(N_neg / N_pos) per threshold (gentler than the full ratio, less "
            "risk of over-correcting into the opposite degenerate). Set None to use uniform β_k=1 "
            "(original Loss-1 behaviour). Apr-30 head-pretrain failure was traced to this: with "
            "c=0 (symmetric) and β_k=1, the imbalanced upper thresholds drove the head's shared "
            "trunk into a constant-output basin within 150 steps."
        },
    )
    corn_bias_init_priors: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Per-threshold marginal prior P(y≥k) for k∈{2,3,4,5}. When set, the CORN "
            "head's final-layer bias is initialized to logit(p_k) so that at step 0 σ(z_k) ≈ p_k "
            "for any input — i.e. the head starts at the data-prevalence point and only has to "
            "learn per-frame deviation. Standard class-imbalance fix (Lin et al. focal-loss §3.3). "
            "With pos_weight enabled but no bias-init, a fresh head with random bias=0 outputs "
            "σ=0.5 at step 0, which is far from the true prior on imbalanced thresholds (e.g. "
            "0.076 for k=5) — the resulting first-step gradient is huge and over-corrects. "
            "Bias-init removes that initial transient. Measured priors (50 batches, 18k-trajectory "
            "Apr-30 train pool): [0.602, 0.399, 0.209, 0.076]."
        },
    )
    # ============================================================================
    # Loss 3 — Failure-rehearsal KL anchor (see losses.md §Loss 3).
    # All three default-off; setting failure_kl_weight>0 AND failure_kl_buffer_size>0
    # activates the anchor on top of whichever progress loss is selected (corn/c51).
    # ============================================================================
    failure_kl_weight: float = field(
        default=0.0,
        metadata={
            "help": "Multiplier λ_kl on the failure-rehearsal KL anchor added to the loss on "
            "SUCCESS-only batches (per losses.md §Loss 3). KL is computed between the current "
            "model's progress logits on a buffered failure input and the snapshot logits stored "
            "when that failure was originally processed. 0.0 (default) disables the anchor with "
            "zero overhead — no buffer is maintained, no KL term is added, behavior is bit-"
            "identical to a vanilla run. Recommended pilot: 0.1; sweep {0.05, 0.1, 0.3} if "
            "first run is surprising."
        },
    )
    failure_kl_buffer_size: int = field(
        default=10,
        metadata={
            "help": "FIFO depth N of the failure-rehearsal buffer (per losses.md §Loss 3). The "
            "buffer is only maintained when failure_kl_weight > 0. With data.stratified_batch_"
            "balance=true post-warmup, every batch is all-success or all-failure; buffer entries "
            "are pushed on failure batches. Retention rule: when buffer < N, peek-only on success "
            "(no discard) so we don't drain scarce snapshots; when buffer ≥ N, pop-and-discard the "
            "oldest after each success step. Steady state at 50/50 stratification: buffer stays at "
            "N, oldest entry is ~N failure-pushes back ≈ 2N wall-steps stale. Memory cost: "
            "~150-300 MB per slot (pixel_values dominate at 16 frames × bf16)."
        },
    )
    failure_kl_apply_when_buffer_below_size: bool = field(
        default=True,
        metadata={
            "help": "If True (default), apply the KL anchor as soon as ANY entry exists in the "
            "buffer; the peek-when-below-N retention rule (see failure_kl_buffer_size) ensures we "
            "don't over-anchor on a single entry by keeping it alive across multiple success steps. "
            "If False, wait until buffer is exactly full before applying — more conservative but "
            "delays anchor activation by O(N) failure-batches at the start of training."
        },
    )
    failure_kl_detach_backbone: bool = field(
        default=False,
        metadata={
            "help": "If True, detach the backbone's hidden states before they enter the head on "
            "the KL re-forward path. KL gradient then flows ONLY through the head (not the LoRA "
            "adapters or — in full FT — the backbone weights). Pros: ~25-40% faster KL re-forward "
            "(skip the heavy backward through the transformer). Cons: anchors only the head, not "
            "the LoRA/backbone interpretation of failures — the model can still drift on failures "
            "via backbone changes. Recommended: False for full fine-tuning (the real target), True "
            "for LoRA smoke tests where speed matters and the backbone is mostly frozen."
        },
    )
    failure_kl_concat_batch: bool = field(
        default=False,
        metadata={
            "help": "If True, concatenate the buffered failure inputs with the success batch along "
            "the batch dim and run ONE forward instead of two. ~15-40% speedup on success steps via "
            "kernel-launch reduction and better GPU utilization. Doubles peak forward-pass memory "
            "(holds 2× activations simultaneously). Default False (sequential) is safe on any "
            "memory configuration; flip True for full FT on H100 80GB where headroom exists."
        },
    )
    failure_kl_apply_every_n_success: int = field(
        default=1,
        metadata={
            "help": "Apply the KL term on every Nth success step (default 1 = every success step). "
            "Increase to halve/third/etc. the wall-clock cost of the anchor at the price of "
            "proportionally weaker effective regularization. With apply_every=2 the per-success-"
            "step KL cost is amortized over two steps, but each KL gradient is applied with full "
            "weight — equivalent to halving the buffer cycling rate, not halving λ_kl."
        },
    )


@dataclass
class SaveBestConfig:
    """Configuration for SaveBestCallback"""

    # Metric monitoring
    metric_names: List[str] = field(
        default_factory=lambda: ["custom_eval/p_rank_spearman_mw"],
        metadata={"help": "List of metric names to monitor for saving best models (will be averaged)"},
    )
    greater_is_better: List[bool] = field(
        default_factory=lambda: [True],
        metadata={"help": "Whether higher values are better for each metric (must match length of metric_names)"},
    )
    keep_top_k: int = field(default=1, metadata={"help": "Number of best checkpoints/uploads to keep"})
    save_every: Optional[int] = field(
        default=None,
        metadata={"help": "Save 'latest' checkpoint every N steps (should be multiple of eval_steps). None disables."},
    )

    # Hub upload configuration
    upload_to_hub: bool = field(default=False, metadata={"help": "Whether to upload best models to HuggingFace Hub"})
    hub_save_every: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frequency (in steps) to upload to Hub. None = upload every checkpoint. Local saves always happen regardless."
        },
    )
    hub_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace token (or set HF_TOKEN env var)"})
    hub_private: bool = field(default=False, metadata={"help": "Whether to make the Hub model private"})

    def __post_init__(self):
        """Validate that metric_names and greater_is_better have the same length"""
        if len(self.metric_names) != len(self.greater_is_better):
            raise ValueError(
                f"metric_names ({len(self.metric_names)}) and greater_is_better ({len(self.greater_is_better)}) must have the same length"
            )


@dataclass
class LoggingConfig:
    """Config for logging settings"""

    save_model: bool = field(default=True)
    save_processor: bool = field(default=True)
    # Logging backends
    log_to: List[str] = field(
        default_factory=list,
        metadata={"help": "List of logging backends to use, e.g., ['wandb', 'tensorboard']"},
    )
    # Wandb configuration
    wandb_project: str = field(default="rbm-model", metadata={"help": "Wandb project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity/username"})
    wandb_notes: Optional[str] = field(
        default=None, metadata={"help": "Optional notes/comment to add to the wandb run"}
    )
    wandb_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Wandb mode: 'online' (default), 'offline' (local files only, no network I/O), or 'disabled'. "
            "Offline mode prevents network I/O blocking that can cause deadlocks in distributed training."
        },
    )
    # Log level: "TRACE", "DEBUG2", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_level: str = field(
        default="INFO",
        metadata={
            "help": "Logging level for console output. Options: TRACE (most verbose), DEBUG2, DEBUG, INFO, WARNING, ERROR, CRITICAL"
        },
    )

    # SaveBest configuration
    save_best: Optional[SaveBestConfig] = field(default=None, metadata={"help": "SaveBestCallback configuration"})


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""

    mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'evaluate'"})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode"})
    trainer_cls: str = field(
        default="rbm_heads",
        metadata={"help": "Trainer class: 'rbm_heads', 'rewind_transformer', 'rewind_scale_transformer'"},
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    custom_eval: CustomEvaluationConfig = field(default_factory=CustomEvaluationConfig)

    def __post_init__(self):
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)

        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)

        if isinstance(self.peft, dict):
            self.peft = PEFTConfig(**self.peft)

        if isinstance(self.data, dict):
            self.data.pop("roboarena_partial_success_threshold", None)
            self.data = DataConfig(**self.data)

        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)

        # Handle nested SaveBestConfig in LoggingConfig
        if hasattr(self.logging, "save_best") and isinstance(self.logging.save_best, dict):
            self.logging.save_best = SaveBestConfig(**self.logging.save_best)
        elif self.logging.save_best is None:
            # Set default SaveBestConfig if not provided
            self.logging.save_best = SaveBestConfig()

        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)
