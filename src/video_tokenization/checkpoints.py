import logging
import os

import torch
import torch.optim as optim

from video_tokenization.tokenizer import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_checkpoint(
    model: VideoTokenizer,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
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
        "config": config.to_dict(),
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
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    device: torch.device,
) -> tuple[VideoTokenizer, optim.Optimizer, optim.lr_scheduler.CosineAnnealingLR]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, optimizer, scheduler
