import logging
import os

import torch
import torch.optim as optim

from video_tokenization.create_tokenizer import create_model
from video_tokenization.model import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_checkpoint(
    model: VideoTokenizer,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_idx: int,
    loss: float,
    config: VideoTokenizerTrainingConfig,
    best_loss: float,
    dataloader_state: dict,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "config": config.__dict__,
        "dataloader_state": dataloader_state,
    }
    checkpoint_path = os.path.join(
        config.checkpoint_dir, f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: VideoTokenizer,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[VideoTokenizer, optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, optimizer, scheduler


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[VideoTokenizer, VideoTokenizerTrainingConfig]:
    """
    Load a VideoTokenizer and its training config from a checkpoint.

    This reconstructs the model using the saved config (including FSQ bins)
    so that quantization levels and related hyperparameters match training.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})

    # Start from default config, then apply saved values
    config = VideoTokenizerTrainingConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = create_model(config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        # Optionally print or log for debugging
        logger.warning(f"Missing keys when loading model checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading model checkpoint: {unexpected}")

    model.to(device)
    model.eval()

    return model, config
