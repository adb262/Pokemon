import logging
import os
from datetime import datetime

import torch
import torch.optim as optim

from dynamics_model.training_args import DynamicsModelTrainingConfig
from dynamics_model.model import DynamicsModel
from latent_action_model.model import LatentActionVQVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


_ORIG_MOD_MARKER = "._orig_mod."


def _strip_orig_mod(key: str) -> str:
    return key.replace(_ORIG_MOD_MARKER, ".")


def adapt_state_dict_to_model(
    state_dict: dict, model: torch.nn.Module
) -> dict:
    """Map checkpoint keys onto the layout the target ``model`` expects.

    ``torch.compile`` wraps submodules in ``OptimizedModule`` (the training
    script compiles ``model.decoder``, ``action_model.encoder``, and
    ``action_model.decoder_transformer``), which inserts ``._orig_mod.`` into
    the wrapped submodule's parameter names. As a result, checkpoints saved
    during training can carry ``._orig_mod.`` in their keys while the
    "canonical" checkpoints (and the uncompiled model used at inference) do
    not.

    To load either flavor into either flavor of model, we canonicalize every
    state-dict key by stripping ``._orig_mod.`` and look up the matching key
    on the model. The model's own ``state_dict`` keys are the source of truth
    for whether ``_orig_mod`` is present, so this adapts in both directions:

    - canonical checkpoint (no ``_orig_mod``) → compiled model: inserts
      ``_orig_mod`` to match the model's wrapped submodule keys.
    - non-canonical checkpoint (with ``_orig_mod``) → uncompiled model:
      strips ``_orig_mod`` to match the canonical key layout.
    - matching layouts on both sides: returned unchanged.
    """
    model_keys = list(model.state_dict().keys())
    if set(state_dict.keys()) == set(model_keys):
        return state_dict

    canonical_to_model_key = {_strip_orig_mod(k): k for k in model_keys}

    adapted: dict = {}
    rewritten = 0
    for key, value in state_dict.items():
        target_key = canonical_to_model_key.get(_strip_orig_mod(key), key)
        if target_key != key:
            rewritten += 1
        adapted[target_key] = value

    if rewritten:
        logger.info(
            "Adapted %d/%d state_dict keys to match model layout "
            "(torch.compile '_orig_mod' canonicalization)",
            rewritten,
            len(state_dict),
        )
    return adapted


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
    action_model: LatentActionVQVAE,
    is_best: bool = False,
    global_step: int | None = None,
):
    """Save comprehensive model checkpoint.

    Writes two files:
      - ``{checkpoint_dir}/checkpoint_epoch{E}_batch{B}.pt`` for the dynamics
        model (also mirrored to ``checkpoint_latest.pt`` and, if ``is_best``,
        ``checkpoint_best.pt``).
      - ``{checkpoint_dir}/action_model/checkpoint_epoch{E}_batch{B}.pt`` for
        the co-trained latent action model, in the same format consumed by
        ``create_action_model_from_dynamics_config`` (key ``model_state_dict``
        at the top level).
    """
    timestamp = datetime.now().isoformat()

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "global_step": global_step,
        "config": config.__dict__,
        "dataloader_state": dataloader_state,
        "timestamp": timestamp,
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

    # Also save the co-trained action model separately so it can be loaded
    # standalone via ``action_model_checkpoint_path``.
    action_checkpoint_dir = os.path.join(config.checkpoint_dir, "action_model")
    os.makedirs(action_checkpoint_dir, exist_ok=True)
    action_checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": action_model.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "global_step": global_step,
        "config": config.__dict__,
        "timestamp": timestamp,
    }
    action_checkpoint_path = os.path.join(
        action_checkpoint_dir, f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"
    )
    torch.save(action_checkpoint, action_checkpoint_path)
    torch.save(
        action_checkpoint, os.path.join(action_checkpoint_dir, "checkpoint_latest.pt")
    )
    if is_best:
        torch.save(
            action_checkpoint,
            os.path.join(action_checkpoint_dir, "checkpoint_best.pt"),
        )
    logger.info(f"Saved action model checkpoint: {action_checkpoint_path}")

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
    state_dict = adapt_state_dict_to_model(checkpoint["model_state_dict"], model)
    model.load_state_dict(state_dict)
    optimizer_state_loaded = False
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_state_loaded = True
    except ValueError as exc:
        logger.warning(
            "Skipping optimizer state load from %s due to parameter-group mismatch: %s",
            checkpoint_path,
            exc,
        )

    if optimizer_state_loaded:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except ValueError as exc:
            logger.warning(
                "Skipping scheduler state load from %s due to state mismatch: %s",
                checkpoint_path,
                exc,
            )
    else:
        logger.warning(
            "Using freshly initialized scheduler because optimizer state was not restored."
        )

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model, optimizer, scheduler, checkpoint
