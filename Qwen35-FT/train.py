import json
import os
from dataclasses import asdict
import shutil

import torch
import torch.distributed as dist
import yaml
from rich import print as rprint
from rich.panel import Panel
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra import main as hydra_main

from peft import prepare_model_for_kbit_training
from robometer.configs.experiment_configs import (
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    DataConfig,
    TrainingConfig,
    LossConfig,
    LoggingConfig,
    SaveBestConfig,
    CustomEvaluationConfig,
)
from robometer.trainers import ReWiNDTrainer, RBMHeadsTrainer
from robometer.data.datasets.helpers import show_available_datasets
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import rank_0_info
from robometer.utils.timer import _timer
from robometer.utils.save import SaveBestCallback, resolve_checkpoint_path, update_cfg_with_pretrained_ckpt
from robometer.utils.setup_utils import (
    create_training_arguments,
    setup_batch_collator,
    setup_dataset,
    setup_model_and_processor,
    setup_peft_model,
)
from robometer.data.datasets.base import resolve_dataset_keys
from robometer.utils.logger import Logger
from robometer.utils.distributed import banner
from robometer.utils.config_utils import display_config, convert_hydra_to_dataclass
import datasets

datasets.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
cs.store(group="model", name="model_config", node=ModelConfig)
cs.store(group="peft", name="peft_config", node=PEFTConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="training", name="training_config", node=TrainingConfig)
cs.store(group="loss", name="loss_config", node=LossConfig)
cs.store(group="logging", name="logging_config", node=LoggingConfig)
cs.store(group="logging/save_best", name="save_best_config", node=SaveBestConfig)
cs.store(group="custom_eval", name="custom_eval_config", node=CustomEvaluationConfig)


import torch

torch.set_num_threads(64)
torch.set_num_interop_threads(8)


def train(cfg: ExperimentConfig):
    timing_raw = {}

    run_name = cfg.training.exp_name
    if cfg.debug:
        run_name += "_debug"
        cfg.training.logging_steps = 1
        cfg.training.eval_steps = 5
        # cfg.data.eval_subset_size = 100
        cfg.training.custom_eval_steps = 5
        cfg.logging.save_best.save_every = 5
        cfg.data.dataloader_num_workers = 0
        cfg.data.dataloader_persistent_workers = False

        # cfg.custom_eval.num_examples_per_quality_pr = 1
        # cfg.custom_eval.policy_ranking_max_tasks = 10

    # Set memory management
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Disambiguate two distinct semantics that previously shared one Hydra knob:
    #   - load_from_checkpoint   = source of MODEL WEIGHTS for setup_model_and_processor.
    #                              Must be an HF-format model (Hub repo or local dir with
    #                              model.safetensors / pytorch_model.bin), because the
    #                              loading path is from_pretrained().
    #   - resume_from_checkpoint = full-state RESUME target for HF Trainer (handled at
    #                              line ~582 via trainer.train(resume_from_checkpoint=...)).
    #                              Supports FSDP SHARDED_STATE_DICT (pytorch_model_fsdp_0/
    #                              with distcp shards) + optimizer + scheduler + RNG.
    # The OLD code was `... or cfg.training.resume_from_checkpoint`, which routed sharded
    # FSDP checkpoint paths through from_pretrained() and crashed with OSError ("no
    # model.safetensors or pytorch_model.bin") because distcp shards aren't a flat HF
    # model. Drop the fall-through: model load comes from load_from_checkpoint (or
    # cfg.model.base_model_id when unset, via setup_model_and_processor's internal default);
    # resume_from_checkpoint stays purely as the HF Trainer resume target.
    checkpoint_to_load = cfg.training.load_from_checkpoint
    if checkpoint_to_load:
        rank_0_info(f"Loading model from checkpoint: {checkpoint_to_load}")
    update_cfg_with_pretrained_ckpt(cfg, checkpoint_to_load)

    # Copy CORN bias-init priors from LossConfig (canonical home) onto ModelConfig so the
    # head construction (which only sees ModelConfig) can read them.
    if getattr(cfg.loss, "corn_bias_init_priors", None) is not None:
        cfg.model.corn_bias_init_priors = list(cfg.loss.corn_bias_init_priors)

    banner("Setting up model and processor")
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw):
        tokenizer, processor, rbm_model = setup_model_and_processor(
            cfg.model,
            hf_model_id=checkpoint_to_load or "",
            peft_config=cfg.peft,
        )

    # Apply PEFT if enabled
    if cfg.model.use_peft:
        peft_rbm_model = setup_peft_model(rbm_model, cfg.peft)
    else:
        peft_rbm_model = rbm_model
        rank_0_info("PEFT not enabled, using full model")

    if cfg.model.quantization:
        peft_rbm_model = prepare_model_for_kbit_training(peft_rbm_model)

    output_dir = os.path.join(cfg.training.output_dir, run_name)

    training_args = create_training_arguments(cfg.training, output_dir)

    # Handle output directory existence (works with accelerate/distributed training)
    overwrite_output_dir = getattr(cfg.training, "overwrite_output_dir", False)

    # Check if distributed training is initialized (for proper synchronization)
    # This is important for accelerate/FSDP setups where multiple processes run
    dist_initialized = dist.is_available() and dist.is_initialized()

    # When resuming, output_dir is EXPECTED to already exist (it holds the checkpoint we are
    # resuming from, e.g. on a SLURM --requeue after a spot preemption). It MUST be preserved:
    # never rmtree it (that would delete the resume checkpoint, then crash on the missing
    # trainer_state.json) and never error on its existence. Overrides overwrite_output_dir for resume.
    resuming_from_checkpoint = bool(getattr(cfg.training, "resume_from_checkpoint", None))

    # Check if output directory exists (only on rank 0 to avoid race conditions)
    if is_rank_0() and os.path.exists(output_dir):
        if resuming_from_checkpoint:
            rank_0_info(
                f"Resuming: output directory {output_dir} already exists — preserving it "
                f"(holds the resume checkpoint; NOT overwriting despite overwrite_output_dir)."
            )
        elif overwrite_output_dir:
            rank_0_info(f"Output directory {output_dir} already exists. Overwriting (overwrite_output_dir=True)...")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                f"Output directory {output_dir} already exists. "
                f"Set overwrite_output_dir=True in config to overwrite it, or use a different output directory."
            )

    # Synchronize all processes before creating directory (important for distributed training)
    # This ensures rank 0 finishes checking/removing before other processes try to create it
    if dist_initialized:
        dist.barrier()

    banner("Creating output directory", f"Logging to: {output_dir}")
    # Create output directory (all processes need to do this for distributed training)
    # os.makedirs is safe to call multiple times (exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Synchronize after directory creation to ensure all processes see it
    if dist_initialized:
        dist.barrier()

    # Initialize logger (works with wandb/tensorboard)
    log_to = cfg.logging.log_to
    log_level = cfg.logging.log_level
    logger = Logger(log_to=log_to, output_dir=output_dir, is_main_process=is_rank_0(), log_level=log_level)
    config_save_path = os.path.join(output_dir, "config.yaml")
    config_dict = asdict(cfg)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    rank_0_info(f"Saved training config to: {config_save_path}")

    # Try to load existing wandb info if resuming training
    wandb_info_path = os.path.join(output_dir, "wandb_info.json")
    resume_id = None
    if os.path.exists(wandb_info_path):
        try:
            with open(wandb_info_path) as f:
                wandb_info = json.load(f)
            resume_id = wandb_info.get("wandb_id")
            if resume_id:
                rank_0_info(f"Found existing wandb run ID: {resume_id}, will resume run")
        except Exception as e:
            rank_0_info(f"Could not load wandb info: {e}")

    # Initialize wandb via logger if requested
    if "wandb" in (cfg.logging.log_to or []) and is_rank_0():
        # Convert config to dict for wandb using dataclass asdict
        config_dict = asdict(cfg)
        logger.init_wandb(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
            config=config_dict,
            notes=cfg.logging.wandb_notes,
            mode=cfg.logging.wandb_mode,
            resume_id=resume_id,
        )
        if resume_id:
            rank_0_info(f"Wandb resumed run: {run_name} (ID: {resume_id})")
        else:
            rank_0_info(f"Wandb initialized: {run_name}")
        if cfg.logging.wandb_notes:
            rank_0_info(f"Wandb notes: {cfg.logging.wandb_notes}")

    logger.write_wandb_info(output_dir, run_name)

    # Use the shared utilities for batch collator and dataset

    if is_rank_0():
        show_available_datasets()

    banner("Resolving dataset keys")
    cfg.data.train_datasets = resolve_dataset_keys(cfg.data.train_datasets, split="train")
    rank_0_info(f"Resolved train datasets: {cfg.data.train_datasets}")

    if cfg.data.eval_datasets:
        cfg.data.eval_datasets = resolve_dataset_keys(cfg.data.eval_datasets, split="eval")
        rank_0_info(f"Resolved eval datasets: {cfg.data.eval_datasets}")

    # Resolve custom evaluation dataset keys once (replace in place)
    for eval_type in cfg.custom_eval.eval_types:
        datasets = getattr(cfg.custom_eval, eval_type, None)
        if datasets:
            resolved = resolve_dataset_keys(datasets, split="eval")
            setattr(cfg.custom_eval, eval_type, resolved)
            rank_0_info(f"Resolved {eval_type} datasets: {resolved}")

    rank_0_info("Dataset keys resolved")

    banner("Setting up training and evaluation datasets and collator")
    with _timer("time/setup_data", timing_raw=timing_raw):
        batch_collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=False)
        train_dataset = setup_dataset(cfg.data)
        num_train_samples = len(train_dataset)
        rank_0_info(f"Training dataset created with {num_train_samples} samples")
        rank_0_info(f"=" * 100)

    # Set up evaluation dataset if evaluation is enabled
    eval_dataset = None
    if cfg.training.do_eval:
        if cfg.data.eval_subset_size is not None:
            dataset_kwargs = {"max_samples": cfg.data.eval_subset_size}
        else:
            dataset_kwargs = {}

        eval_dataset = setup_dataset(cfg.data, is_eval=True, **dataset_kwargs)
        num_eval_samples = len(eval_dataset)
        rank_0_info(f"Evaluation dataset created with {num_eval_samples} samples")

    banner("Setting up trainer", f"Trainer class: {cfg.trainer_cls}")
    trainer_cls = {
        "rbm_heads": RBMHeadsTrainer,
        "rewind_transformer": ReWiNDTrainer,
        "rewind_scale_transformer": ReWiNDTrainer,
    }[cfg.trainer_cls]

    # Add SaveBestCallback to automatically save and upload best models
    save_best_cfg = cfg.logging.save_best
    save_callback = SaveBestCallback(
        **asdict(save_best_cfg),
        base_model=cfg.model.base_model_id,
    )

    trainer = trainer_cls(
        model=peft_rbm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=batch_collator,
        config=cfg,
        logger=logger,
        callbacks=[save_callback],
    )

    # Set trainer reference in the callback so it can access trainer methods
    save_callback.setup_trainer_reference(trainer)

    # Debug: Check if callback was added
    rank_0_info(f"🔧 DEBUG: Trainer callbacks: {[type(cb).__name__ for cb in trainer.callback_handler.callbacks]}")

    metrics_info = []
    for name, is_better in zip(save_best_cfg.metric_names, save_best_cfg.greater_is_better):
        direction = "↗️ higher" if is_better else "↘️ lower"
        metrics_info.append(f"{name} ({direction})")

    rank_0_info(f"💾 SaveBest monitoring: {', '.join(metrics_info)}")
    rank_0_info(f"📁 Keeping top {save_best_cfg.keep_top_k} checkpoint(s) and upload(s)")

    if is_rank_0():
        print("\n" + "=" * 80)
        print("--- PRE-TRAINING FSDP DIAGNOSTICS ---")
        # The Trainer creates its own Accelerator instance. Let's check its state.
        if hasattr(trainer, "accelerator"):
            print("Trainer's Accelerator object found.")
            fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin:
                print("FSDP Plugin found in Accelerator state.")
                # This is the configuration the accelerator will ACTUALLY use for wrapping.
                print(f"VERIFY: Actual FSDP plugin config being used: {fsdp_plugin}")
            else:
                print("ERROR: FSDP Plugin NOT found in the Trainer's accelerator state!")
        else:
            print("ERROR: Trainer has no 'accelerator' attribute yet. This check needs to be later.")
        print("=" * 80 + "\n")

    # log timing_raw via logger
    if is_rank_0():
        logger.log_scalars(timing_raw)

    rank_0_info(f"Timing raw: {timing_raw}")

    # Full resume: restore optimizer state and step counter (load_from_checkpoint only loads weights at setup)
    hub_token = (save_best_cfg.hub_token if save_best_cfg else None) or os.environ.get("HF_TOKEN")
    resume_path = (
        resolve_checkpoint_path(cfg.training.resume_from_checkpoint, hub_token=hub_token)
        if cfg.training.resume_from_checkpoint
        else None
    )
    if resume_path:
        rank_0_info(f"Resuming training from checkpoint: {resume_path}")
    else:
        rank_0_info("Training from step 0 (no resume)")

    # Restore random state from checkpoint only when doing full resume
    if resume_path and os.path.isdir(resume_path):
        random_state_file = os.path.join(resume_path, "dataset_random_state.json")
        if os.path.exists(random_state_file):
            try:
                with open(random_state_file, "r") as f:
                    random_state = json.load(f)
                # Handle RepeatedDataset wrapper if present
                train_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
                if hasattr(train_dataset, "set_random_state"):
                    train_dataset.set_random_state(random_state)
                    rank_0_info(f"Restored dataset random state from {random_state_file}")
                else:
                    rank_0_info(f"Dataset does not support random state restoration")
            except Exception as e:
                rank_0_info(f"Could not restore random state: {e}")
        else:
            rank_0_info(f"No dataset_random_state.json found in checkpoint, starting with fresh random state")

    if cfg.debug:
        rank_0_info("🐛 DEBUG MODE: eval_steps=2, custom_eval_steps=2, eval_subset_size=10")

    if cfg.mode == "evaluate":
        # Eval-only path: skip the training loop entirely. trainer.evaluate() prepares the
        # model via accelerator, runs the standard eval, and ends by calling
        # _run_custom_evaluations(), which dumps policy_ranking JSONs per dataset.
        # Used for the held-out test set comparison (baseline vs L1 vs L2).
        rank_0_info("Eval-only mode — skipping training, running evaluation only.")
        if resume_path:
            # Manually load weights since we're not going through trainer.train()'s resume path.
            from transformers.trainer_utils import get_last_checkpoint
            ckpt = resume_path if os.path.isdir(resume_path) else get_last_checkpoint(resume_path)
            rank_0_info(f"Loading weights from {ckpt} for eval-only mode")
            trainer._load_from_checkpoint(ckpt)
        trainer.evaluate()
        rank_0_info(f"Eval complete! See {output_dir}/eval_results for per-dataset JSON dumps.")
        return

    if cfg.mode == "debug_corn":
        # CORN-pretrain debug path: load model + dataloader, run 1 batch, dump:
        #   (1) gradient stats on the CORN head's parameters (sanity check the loss path)
        #   (2) cosine similarity of per-frame hidden states within/across trajectories
        #       (sanity check whether the frozen backbone carries per-frame discrimination)
        #   (3) per-batch threshold-positive rates (verify the dataloader yields the
        #       distribution we computed offline in CORN_PRETRAIN_DEBUG.md)
        # Outputs a `corn_debug.json` next to training.output_dir for post-mortem.
        import json as _json
        import numpy as _np
        import torch.nn.functional as _F
        rank_0_info("=" * 80)
        rank_0_info("CORN-pretrain DEBUG mode — running 1-batch diagnostic")
        rank_0_info("=" * 80)

        # IMPORTANT: do NOT call accelerator.prepare_model on the model — the model
        # returns (output, timing_raw) where timing_raw is a defaultdict, and accelerate's
        # convert_to_fp32 hook (added by prepare_model when mixed_precision='no') tries to
        # rebuild the dict via `type(data)(generator)`, which fails for defaultdict because
        # its first arg must be a callable factory. Bypass accelerate entirely; we just want
        # one forward+backward on a single GPU. Initialize the accelerator (some trainer code
        # references self.accelerator.device) but never wrap the model.
        trainer.create_accelerator_and_postprocess()
        device = trainer.accelerator.device
        model = trainer.model
        model.to(device)
        model.train()
        train_loader = trainer.get_train_dataloader()

        # ---- (3) Per-batch threshold-positive rates over 50 batches ----
        # Each batch is a nested dict with `progress_inputs` containing target_progress and
        # target_progress_mask. The trainer's _compute_loss path digs in the same way.
        rank_0_info("\n[3] Sampling 50 batches for effective per-threshold positive rates …")
        batch_iter = iter(train_loader)
        per_batch_stats = []
        n_examined = 0
        for i in range(50):
            try:
                b = next(batch_iter)
            except StopIteration:
                break
            pi = b.get("progress_inputs") or {}
            tp = pi.get("target_progress")
            if tp is None:
                continue
            # Match the trainer's CORN encoding: y = round(tp*4) + 1, then clamp to {1..5}.
            # Earlier version of this script omitted the +1 shift, which made the reported
            # P(y≥k) numbers correspond to k+1 in the trainer's space — the false "P(y≥5)=0"
            # was actually P(y≥4) misidentified.
            y = ((tp.float() * 4.0).round().long() + 1).clamp(1, 5)       # [B, T]
            thresh = torch.arange(2, 6, device=y.device)
            bits = (y.unsqueeze(-1) >= thresh).float()                    # [B, T, 4]
            # Skip masking for the distribution diagnostic — small fraction of frames
            # are masked (padding / partial labels), and the label distribution is what
            # matters here, not the exact loss-weighted distribution.
            denom = bits.shape[0] * bits.shape[1] + 1e-9
            per_batch_stats.append((bits.sum(dim=(0, 1)) / denom).tolist())
            n_examined += 1

        rank_0_info(f"  examined {n_examined} batches with progress_inputs")
        if not per_batch_stats:
            rank_0_info("  (NO batches had progress_inputs — sampler may be emitting only preference samples)")
            thresh_means = [float('nan')] * 4
            thresh_stds = [float('nan')] * 4
            pos_weights_sqrt = [1.0, 1.0, 1.0, 1.0]
            pos_weights_full = [1.0, 1.0, 1.0, 1.0]
        else:
            per_batch_arr = _np.array(per_batch_stats)
            thresh_means = per_batch_arr.mean(axis=0).tolist()
            thresh_stds  = per_batch_arr.std(axis=0).tolist()
            for ki, k in enumerate((2, 3, 4, 5)):
                rank_0_info(f"  threshold k={k}: P(y≥k) mean={thresh_means[ki]:.4f}  std={thresh_stds[ki]:.4f}")
            # Recommended pos_weights — sqrt(N_neg / N_pos) per Apr-30 plan
            pos_weights_full = [(1 - p) / max(p, 1e-6) for p in thresh_means]
            pos_weights_sqrt = [(p_w) ** 0.5 for p_w in pos_weights_full]
            rank_0_info(f"  → recommended pos_weights (sqrt formula): {[f'{w:.3f}' for w in pos_weights_sqrt]}")
            rank_0_info(f"    (full ratio for reference): {[f'{w:.3f}' for w in pos_weights_full]}")

        # ---- (1) + (2) + a forward+backward on a single batch ----
        rank_0_info("\n[1+2] Single forward+backward on one batch …")
        # Reset iterator and grab one batch fresh
        batch_iter = iter(train_loader)
        single_batch = next(batch_iter)

        # Capture the per-prog-token hidden states that the head sees, by hooking
        # _extract_hidden_state_from_token's output. We tag onto a list during the next
        # forward pass.
        prog_hidden_capture = []
        rbm = trainer.model
        # Find the underlying RBM (might be wrapped in PEFT/DDP/etc.)
        rbm_inner = rbm
        for attr in ("module", "model"):
            if hasattr(rbm_inner, attr) and not isinstance(getattr(rbm_inner, attr), torch.nn.Linear):
                next_inner = getattr(rbm_inner, attr)
                if hasattr(next_inner, "_extract_hidden_state_from_token"):
                    rbm_inner = next_inner
                    break

        original_extract = None
        if hasattr(rbm_inner, "_extract_hidden_state_from_token"):
            # Capture by replacing the bound method on the instance with a closure that
            # delegates to the original bound method and tees the output into our list
            # when the token is <|prog_token|>. The original is already bound, so we don't
            # need to re-bind — just call it.
            original_extract = rbm_inner._extract_hidden_state_from_token
            def _capture(hidden_state, input_ids, token):
                out = original_extract(hidden_state, input_ids, token)
                if token == "<|prog_token|>":
                    prog_hidden_capture.append(out)
                return out
            rbm_inner._extract_hidden_state_from_token = _capture

        # Bypass trainer.compute_loss → forward_model wrapping by calling the raw model
        # forward directly. compute_loss goes through accelerate which fails on defaultdict
        # in the model's return tuple (see comment above near prepare_model).
        # Move tensors to device (recurses into nested dicts/lists/tuples).
        # Use collections.abc.Mapping so we handle transformers' BatchFeature/BatchEncoding
        # (UserDict subclasses, NOT dict — `isinstance(bf, dict)` is False).
        from collections.abc import Mapping as _Mapping
        def _to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, _Mapping):
                return {k: _to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_device(v, device) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_device(v, device) for v in obj)
            return obj
        single_batch = _to_device(single_batch, device)

        progress_inputs = single_batch.get("progress_inputs", {})
        if not progress_inputs:
            rank_0_info("  ERROR: batch has no progress_inputs — sampler is emitting only preference samples")
            loss = torch.tensor(0.0, device=device)
            head_grads = {}
        else:
            # Forward through the raw (unwrapped) model. forward_model just calls model(**kwargs)
            # and unpacks (output, timing_raw); no accelerate fp32 conversion involved.
            model_output, _ = trainer.forward_model(model, progress_inputs, sample_type="progress")
            progress_logits = model_output.progress_logits
            progress_pred = progress_logits["A"]
            progress_target = progress_inputs["target_progress"]
            progress_target_mask = progress_inputs["target_progress_mask"].unsqueeze(-1)
            predict_last_frame_mask = progress_inputs.get("predict_last_frame_mask", None)
            loss, _spearman, _metrics = trainer._compute_progress_loss_helper(
                progress_pred, progress_target, progress_target_mask,
                predict_last_frame_mask=predict_last_frame_mask,
            )
            rank_0_info(f"  forward loss = {loss.item():.6f}")
            loss.backward()

        # Restore extract hook
        if original_extract is not None:
            rbm_inner._extract_hidden_state_from_token = original_extract

        # ---- (1) Gradient stats on CORN head ----
        rank_0_info("\n[1] CORN head gradient stats")
        head_grads = {}
        # Find progress_head module
        for n, p in model.named_parameters():
            if "progress_head" in n and p.grad is not None:
                g = p.grad
                head_grads[n] = {
                    "shape": list(g.shape),
                    "abs_mean": g.abs().mean().item(),
                    "abs_max":  g.abs().max().item(),
                    "n_zero":   int((g == 0).sum().item()),
                    "n_total":  int(g.numel()),
                    "has_nan":  bool(torch.isnan(g).any().item()),
                }
                rank_0_info(
                    f"  {n}  shape={list(g.shape)}  |grad|_mean={g.abs().mean().item():.3e}  "
                    f"|grad|_max={g.abs().max().item():.3e}  zeros={int((g == 0).sum().item())}/{int(g.numel())}"
                )

        # ---- (2) Cosine sim of per-prog-token hidden states ----
        rank_0_info("\n[2] Cosine similarity of per-prog-token hidden states")
        within_traj_sims = []
        across_traj_sims = []
        within_flat = []
        if prog_hidden_capture:
            tokens = prog_hidden_capture[0]
            # tokens is a list of tensors, one per sample; each is [n_tokens, hidden_dim]
            if isinstance(tokens, list) and len(tokens) > 0:
                # Within-trajectory: pairwise cosine within each sample's token sequence
                for t in tokens:
                    if t.shape[0] < 2: continue
                    norm_t = _F.normalize(t.float(), dim=-1)
                    cos = (norm_t @ norm_t.T)               # [n_tok, n_tok]
                    upper = cos[torch.triu_indices(cos.shape[0], cos.shape[1], offset=1).unbind()]
                    within_traj_sims.append(upper.cpu().tolist())
                # Across-trajectory: cosine between each sample's LAST token
                last_tokens = torch.stack([_F.normalize(t.float(), dim=-1)[-1] for t in tokens if t.shape[0] > 0])
                across_cos = (last_tokens @ last_tokens.T)
                upper = across_cos[torch.triu_indices(across_cos.shape[0], across_cos.shape[1], offset=1).unbind()]
                across_traj_sims = upper.cpu().tolist()
            within_flat = [v for sub in within_traj_sims for v in sub]
            if within_flat:
                wm = float(_np.mean(within_flat))
                rank_0_info(f"  within-trajectory cosine: mean={wm:.4f}  std={_np.std(within_flat):.4f}  "
                            f"(healthy: 0.80–0.95; collapsed: >0.99)")
            if across_traj_sims:
                am = float(_np.mean(across_traj_sims))
                rank_0_info(f"  across-trajectory cosine (last-token): mean={am:.4f}  std={_np.std(across_traj_sims):.4f}  "
                            f"(healthy: <0.85; collapsed: >0.99)")
        else:
            rank_0_info("  (could not capture per-prog-token hidden states — hook didn't fire)")

        # Save report to JSON next to output_dir
        report_path = os.path.join(output_dir, "corn_debug.json")
        report = {
            "loss": float(loss.item()) if loss is not None else None,
            "head_grads": head_grads,
            "thresh_positive_rates_mean": thresh_means,
            "thresh_positive_rates_std": thresh_stds,
            "pos_weights_sqrt": pos_weights_sqrt,
            "pos_weights_full": pos_weights_full,
            "within_traj_cosine_mean": float(_np.mean(within_flat)) if within_flat else None,
            "across_traj_cosine_mean": float(_np.mean(across_traj_sims)) if across_traj_sims else None,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            _json.dump(report, f, indent=2)
        rank_0_info(f"\n→ Wrote report to {report_path}")
        return

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(cfg.training.output_dir)
    rank_0_info(f"Training complete! Check {cfg.training.output_dir} for checkpoints and final model.")


@hydra_main(version_base=None, config_path="robometer/configs", config_name="config")
def main(cfg: DictConfig):
    banner("Starting Robometer Training")

    # Convert Hydra config to dataclass
    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    # Display the configuration in a nice Rich format
    display_config(exp_cfg)

    if exp_cfg.mode == "train":
        if is_rank_0():
            rprint(Panel.fit("🚀 Starting Robometer Training", style="bold green"))
        train(exp_cfg)
    elif exp_cfg.mode == "evaluate":
        if is_rank_0():
            rprint(Panel.fit("📊 Eval-only mode (no training)", style="bold cyan"))
        train(exp_cfg)  # train() now early-returns when cfg.mode == 'evaluate'
    elif exp_cfg.mode == "debug_corn":
        if is_rank_0():
            rprint(Panel.fit("🐛 CORN-pretrain DEBUG (one-batch diagnostic)", style="bold yellow"))
        train(exp_cfg)
    else:
        raise ValueError(f"Unknown mode: {exp_cfg.mode}. Must be 'train', 'evaluate', or 'debug_corn'")


if __name__ == "__main__":
    main()
