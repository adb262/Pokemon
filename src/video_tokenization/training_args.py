# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

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
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    num_epochs: int = 5
    device: str = "mps" if torch.backends.mps.is_available() else "cuda"
    frames_dir: str = "pokemon_frames"
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    seed: int = 42
    num_images_in_video: int = 2
    warmup_steps: int = 100
    d_model: int = 256
    num_transformer_layers: int = 4
    latent_dim: int = 64
    fsq_latent_dim: int = 1
    num_embeddings: int = 8
    num_heads: int = 2
    resume_from: Optional[str] = None
    experiment_name: Optional[str] = None
    seed_cache: bool = False
    # Wandb Configuration
    use_wandb: bool = True
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
    # Temporary attributes for S3 operations
    _temp_log_file: Optional[str] = None
    _temp_tensorboard_dir: Optional[str] = None
