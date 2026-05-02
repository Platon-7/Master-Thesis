#!/usr/bin/env python3
"""
Shared prediction heads plugin for RBM and ReWiND models.
Provides a mixin class that models can inherit from to get prediction heads.
"""

import torch
import torch.nn as nn
from typing import Optional
from robometer.utils.logger import get_logger

logger = get_logger()


class PredictionHeadsMixin(nn.Module):
    """
    Mixin class that provides prediction heads for reward models.

    Models should inherit from this mixin and pass hidden_dim and optionally model_config
    via **kwargs to super().__init__().
    """

    def __init__(
        self,
        *args,
        hidden_dim: Optional[int] = None,
        model_config: Optional[object] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize prediction heads if parameters are provided.

        Args:
            hidden_dim: Input hidden dimension for all heads (if None, heads won't be initialized)
            model_config: Optional model config object with loss settings (if provided, extracts progress settings from it)
            dropout: Dropout rate for all heads
        """
        super().__init__(*args, **kwargs)

        if hidden_dim is not None:
            # Extract progress settings from model_config if provided
            if model_config is not None:
                progress_output_size = 1  # Default: continuous output
                progress_use_sigmoid = True
                use_discrete_progress = False
                # Check for progress_loss_type in model_config (direct attribute or under loss)
                progress_loss_type = None
                if hasattr(model_config, "progress_loss_type"):
                    progress_loss_type = model_config.progress_loss_type
                elif hasattr(model_config, "loss") and hasattr(model_config.loss, "progress_loss_type"):
                    progress_loss_type = model_config.loss.progress_loss_type

                # Detect the CORN head variant (losses.md §Loss 1). When active, the progress head
                # emits 4 cumulative-threshold logits (one per k in {2,3,4,5}) without sigmoid —
                # sigmoid is applied inside the loss for numerical stability (F.logsigmoid).
                progress_head_mode = getattr(model_config, "progress_head_mode", None)
                if progress_head_mode is None and hasattr(model_config, "model"):
                    progress_head_mode = getattr(model_config.model, "progress_head_mode", None)
                progress_head_mode = (progress_head_mode or "c51").lower()

                if progress_head_mode == "corn":
                    # CORN: 4 threshold logits, no sigmoid, decoupled from progress_discrete_bins.
                    progress_output_size = 4
                    progress_use_sigmoid = False
                    use_discrete_progress = True  # Still "not-continuous" — downstream shape is [B, T, 4]
                elif progress_loss_type and progress_loss_type.lower() in ("discrete", "c51_asymmetric"):
                    use_discrete_progress = True
                    if hasattr(model_config, "progress_discrete_bins"):
                        progress_output_size = model_config.progress_discrete_bins
                    elif hasattr(model_config, "loss") and hasattr(model_config.loss, "progress_discrete_bins"):
                        progress_output_size = model_config.loss.progress_discrete_bins
                    else:
                        progress_output_size = 10  # Default bins
                    progress_use_sigmoid = False
                # Set use_discrete_progress attribute for models to use
                self.use_discrete_progress = use_discrete_progress
                self.progress_head_mode = progress_head_mode
            else:
                # Default values if no model_config
                progress_output_size = 1
                progress_use_sigmoid = True
                self.use_discrete_progress = False

            logger.info(f"PredictionHeadsMixin. __init__: use_discrete_progress: {self.use_discrete_progress}")

            # Progress head
            progress_layers = [
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, progress_output_size),
            ]
            if progress_use_sigmoid:
                progress_layers.append(nn.Sigmoid())
            self.progress_head = nn.Sequential(*progress_layers)

            # CORN bias-init: when corn_bias_init_priors is set in loss config, initialize
            # the final-layer bias to logit(p_k) so σ(z_k) ≈ p_k at step 0 for any input.
            # Standard class-imbalance fix — without it, a random-init head outputs σ=0.5
            # at step 0 which is wildly off the prior on imbalanced thresholds (e.g. 0.076
            # for k=5), producing huge first-step gradients that destabilize training.
            if progress_head_mode == "corn":
                bias_priors = None
                # Try ModelConfig first (mirror field, populated by train.py from LossConfig);
                # fall back to ExperimentConfig.loss path for callers that pass the full cfg.
                if hasattr(model_config, "corn_bias_init_priors"):
                    bias_priors = model_config.corn_bias_init_priors
                if bias_priors is None and hasattr(model_config, "loss") and hasattr(model_config.loss, "corn_bias_init_priors"):
                    bias_priors = model_config.loss.corn_bias_init_priors
                if bias_priors is not None and len(bias_priors) == 4:
                    final_linear = progress_layers[-1]  # last appended layer is the 4-out Linear
                    if isinstance(final_linear, nn.Linear) and final_linear.out_features == 4:
                        with torch.no_grad():
                            priors = torch.tensor(list(bias_priors), dtype=final_linear.bias.dtype)
                            priors = priors.clamp(min=1e-4, max=1.0 - 1e-4)
                            logits = torch.log(priors / (1.0 - priors))  # logit(p_k)
                            final_linear.bias.data.copy_(logits)
                        logger.info(
                            f"CORN head: bias-initialized final layer to logit(priors) = "
                            f"{[round(x, 3) for x in logits.tolist()]} from priors {list(bias_priors)}"
                        )

            # Preference head
            self.preference_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Success head
            self.success_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
