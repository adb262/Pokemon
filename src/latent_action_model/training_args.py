# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class VideoTrainingConfig:
    image_size: int = 256
    patch_size: int = 8
    batch_size: int = 1
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
    d_model: int = 256
    num_transformer_layers: int = 4
    latent_dim: int = 64
    num_embeddings: int = 8
    num_heads: int = 2
    resume_from: Optional[str] = None
    experiment_name: Optional[str] = None
    seed_cache: bool = False
    # Wandb Configuration
    use_wandb: bool = True
    wandb_project: str = "pokemon-action-vqvae"
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
    save_dir: str = "latent_action_results"
    # Temporary attributes for S3 operations
    _temp_log_file: Optional[str] = None
    _temp_tensorboard_dir: Optional[str] = None

    @classmethod
    def from_cli(cls) -> "VideoTrainingConfig":
        parser = argparse.ArgumentParser(description="Pokemon VQVAE Training")
        parser.add_argument(
            "--frames_dir", type=str, help="Frames directory (local path or S3 prefix)"
        )
        parser.add_argument(
            "--resume", type=str, help="Path to checkpoint to resume from"
        )
        parser.add_argument(
            "--experiment_name", type=str, help="Name for this experiment"
        )
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument("--num_epochs", type=int, help="Number of epochs")
        parser.add_argument(
            "--num_images_in_video", type=int, help="Number of frames in video"
        )
        parser.add_argument(
            "--use_s3", type=str, choices=["true", "false"], help="Use S3 for storage"
        )
        parser.add_argument(
            "--cache_dir", type=str, help="Local cache directory for S3 images"
        )
        parser.add_argument(
            "--seed_cache", action="store_true", help="Seed cache with all frames"
        )
        parser.add_argument(
            "--save_dir", type=str, help="Directory to save visualizations and grids"
        )

        # Wandb arguments
        parser.add_argument(
            "--use_wandb",
            action="store_true",
            default=True,
            help="Use Weights & Biases for logging",
        )
        parser.add_argument(
            "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            help="Wandb project name",
        )
        parser.add_argument("--wandb_entity", type=str, help="Wandb entity/team name")
        parser.add_argument(
            "--wandb_tags", type=str, nargs="*", help="Wandb tags for the run"
        )
        parser.add_argument("--wandb_notes", type=str, help="Notes for the wandb run")
        parser.add_argument(
            "--save_interval", type=int, help="Interval to save results"
        )

        args = parser.parse_args()

        if args.use_s3 == "true":
            args.use_s3 = True
            args.s3_bucket = os.getenv("S3_BUCKET_NAME")
            args.s3_region = os.getenv("AWS_REGION", "us-east-1")

        return cls.from_args(args)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    def to_dict(self):
        return self.__dict__.copy()

    @staticmethod
    def from_args(args: argparse.Namespace) -> "VideoTrainingConfig":
        default_config = VideoTrainingConfig()
        for key, value in vars(args).items():
            if value is not None:
                setattr(default_config, key, value)

        return default_config
