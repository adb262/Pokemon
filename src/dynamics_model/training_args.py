import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


@dataclass
class DynamicsModelTrainingConfig:
    mask_ratio_lower_bound: float = 0.5
    mask_ratio_upper_bound: float = 1.0
    gradient_clipping: float | None = 1.0
    image_size: int = 128
    patch_size: int = 4
    num_images_in_video: int = 5
    frame_spacing: int = 1
    num_unique_frames: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    warmup_steps: int = 100
    min_learning_rate: float = 1e-6
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 1000
    checkpoint_dir: str = "dynamics_model_checkpoints"
    save_dir: str = "dynamics_model_results"

    # Action model configuration
    action_bins: list[int] = field(default_factory=lambda: [6, 4])
    action_d_model: int = 256
    action_num_transformer_layers: int = 4
    action_num_heads: int = 2
    action_latent_dim: int = 64

    # Dynamics model configuration
    dynamics_d_model: int = 512
    dynamics_num_transformer_layers: int = 8
    dynamics_num_heads: int = 8

    # When True, the latent-action model's decoder output is interpreted as
    # frame-to-frame residuals (and we reconstruct next frames as
    # ``prev + residual``). When False, the decoder output is treated as the
    # next frames directly and residual-specific plots/metrics are skipped.
    predict_action_residuals: bool = False
    action_decoder_loss: Literal["l2", "clipped_l2"] = "l2"
    action_l2_clip_c: float = 10.0
    dynamics_token_loss: Literal["ce", "clipped_ce"] = "ce"
    dynamics_ce_clip_c: float = 0.03

    action_learning_rate: float = 5e-5
    dynamics_learning_rate: float = 1e-4

    # Pretrained tokenizer
    tokenizer_checkpoint_path: str = ""

    # Wandb Configuration
    use_wandb: bool = True
    logging_backend: Literal["wandb", "tensorboard", "none"] = "wandb"
    tensorboard_dir: str = "runs"
    wandb_project: str = "pokemon-dynamics-model"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    wandb_notes: Optional[str] = None

    action_model_checkpoint_path: Optional[str] = None
    dynamics_model_checkpoint_path: Optional[str] = None

    local_cache_dir: Optional[str] = os.environ.get("BT_RW_CACHE_DIR", "cache")
    max_cache_size: int = 100000
    frames_dir: str = "pokemon_frames"
    use_s3: bool = False
    dataset_train_key: Optional[str] = None
    sync_from_s3: bool = False
    experiment_name: Optional[str] = None
    seed: int = 42
    dataset_type: Literal["pokemon", "atari_pong"] = "pokemon"
    atari_pong_data_dir: Optional[str] = None
    atari_pong_crop_scoreboard: bool = False
    atari_pong_require_full_gameplay: bool = False
    dataset_limit: int = 500000

    # Rollout evaluation
    rollout_max_steps: int = 5
    rollout_eval_batches: int = 1
    # Run rollout eval only once every N eval-suite invocations, i.e. every
    # ``eval_interval * rollout_every_n_evals`` optimizer steps. Standard
    # (non-rollout) eval still runs every ``eval_interval`` steps. Defaults
    # to 1, which matches the previous behavior of rolling out on every eval.
    rollout_every_n_evals: int = 5

    # Performance optimizations
    scheduled_sampling: Literal["bengio_per_frame", "free_run_mix", "off"] = "bengio_per_frame"
    use_bf16: bool = True
    use_compile: bool = True
