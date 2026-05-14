# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class VideoTokenizerTrainingConfig:
    dataset_train_key: Optional[str] = None
    sync_from_s3: bool = False
    image_size: int = 256
    patch_size: int = 8
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-7
    num_epochs: int = 5
    device: str = "mps" if torch.backends.mps.is_available() else "cuda"
    frames_dir: str = "pokemon_frames"
    log_interval: int = 10
    save_interval: int = 5000
    eval_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    seed: int = 42
    num_images_in_video: int = 2
    warmup_steps: int = 100
    d_model: int = 256
    num_transformer_layers: int = 4
    latent_dim: int = 64
    num_embeddings: int = 8
    num_heads: int = 2
    experiment_name: Optional[str] = None
    seed_cache: bool = False
    # Wandb Configuration
    use_wandb: bool = True
    logging_backend: Literal["wandb", "tensorboard", "none"] = "wandb"
    wandb_project: str = "pokemon-vqvae"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    wandb_notes: Optional[str] = None
    # S3 Configuration
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_checkpoint_prefix: str = "baseten_test_checkpoints"
    s3_tensorboard_prefix: str = "baseten_test_tensorboard"
    s3_logs_prefix: str = "baseten_test_logs"
    local_cache_dir: Optional[str] = os.environ.get("BT_RW_CACHE_DIR", "cache")
    max_cache_size: int = 100000
    no_wandb: bool = False
    bins: list[int] = field(default_factory=lambda: [8, 8, 6, 5])
    save_dir: str = "tokenization_results"
    max_comparison_images: int = 5
    max_comparison_frames: int = 5
    reconstruction_error_scale: float = 5.0
    # Reconstruction loss configuration. ``l2`` is the standard mean-squared
    # error between the input video and its reconstruction. ``clipped_l2``
    # floors the per-pixel MSE at ``l2_clip_c`` (interpreted in 0-255 pixel
    # units, then rescaled to [0, 1]) so easy-to-fit pixels can't push the
    # average loss arbitrarily close to zero. Mirrors the action-decoder
    # loss option in ``DynamicsModelTrainingConfig``.
    reconstruction_loss_type: Literal["l2", "clipped_l2"] = "l2"
    l2_clip_c: float = 10.0
    dataset_type: Literal["pokemon", "atari_pong"] = "pokemon"
    atari_pong_data_dir: Optional[str] = None
    atari_pong_crop_scoreboard: bool = False
    atari_pong_require_full_gameplay: bool = False
    frame_spacing: int = 1
    num_unique_frames: Optional[int] = None
    dataset_limit: int = 50000
    # Cap on the number of windows kept in the held-out test split. Eval runs
    # iterate the entire test loader on every call, so this controls how many
    # samples each eval step measures against. ``None`` keeps the full split.
    test_dataset_limit: Optional[int] = 1000
    # Early stopping (evaluation-level, not epoch-level). Counts evaluation
    # events — both periodic in-epoch evals (every ``eval_interval``
    # optimizer steps) and the post-epoch eval. Training stops when this
    # many consecutive evals fail to improve eval loss by more than
    # ``early_stopping_min_delta``. ``patience <= 0`` disables early stopping.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    # Performance optimizations
    scheduled_sampling: Literal["bengio_per_frame", "free_run_mix", "off"] = "bengio_per_frame"
    use_bf16: bool = True
    use_compile: bool = True

    # Temporary attributes for S3 operations
    _temp_log_file: Optional[str] = None
    _temp_tensorboard_dir: Optional[str] = None

    def comparison_frame_count(self) -> int:
        """Number of frames to render in train/eval comparison grids."""
        return min(self.max_comparison_frames, self.num_images_in_video)
