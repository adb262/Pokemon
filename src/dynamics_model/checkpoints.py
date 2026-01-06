import logging
import os
from datetime import datetime

import torch
import torch.optim as optim

from dynamics_model.training_args import DynamicsModelTrainingConfig
from dynamics_model.model import DynamicsModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: DynamicsModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_idx: int,
    loss: float,
    config: DynamicsModelTrainingConfig,
    best_loss: float,
    dataloader_state: dict,
    is_best: bool = False,
):
    """Save comprehensive model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "config": config.__dict__,
        "dataloader_state": dataloader_state,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        config.checkpoint_dir, f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    # Save best checkpoint if this is the best
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"New best checkpoint saved: {best_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: DynamicsModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[DynamicsModel, optim.Optimizer, optim.lr_scheduler.LRScheduler, dict]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model, optimizer, scheduler, checkpoint
