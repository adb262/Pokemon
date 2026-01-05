import os
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DynamicsModelTrainingConfig:
    mask_ratio_lower_bound: float = 0.5
    mask_ratio_upper_bound: float = 1.0
    gradient_clipping: float | None = None
    image_size: int = 128
    patch_size: int = 4
    num_images_in_video: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 20
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
    save_interval: int = 500
    eval_interval: int = 1000
    checkpoint_dir: str = "dynamics_model_checkpoints"
    save_dir: str = "dynamics_model_results"

    # Action model configuration
    action_bins: list[int] = field(default_factory=lambda: [8])
    action_d_model: int = 256
    action_num_transformer_layers: int = 4
    action_num_heads: int = 2
    action_latent_dim: int = 64

    # Dynamics model configuration
    dynamics_d_model: int = 512
    dynamics_num_transformer_layers: int = 12
    dynamics_num_heads: int = 8

    action_learning_rate: float = 1e-4
    dynamics_learning_rate: float = 1e-4

    # Pretrained tokenizer
    tokenizer_checkpoint_path: str = ""

    # Wandb Configuration
    use_wandb: bool = True
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
