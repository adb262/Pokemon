import logging
import os

import torch
import torch.optim as optim

from video_tokenization.create_tokenizer import create_model
from video_tokenization.model import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_ORIG_MOD_MARKER = "._orig_mod."
_ORIG_MOD_PREFIX = "_orig_mod."


def _strip_orig_mod(key: str) -> str:
    if key.startswith(_ORIG_MOD_PREFIX):
        key = key[len(_ORIG_MOD_PREFIX):]
    return key.replace(_ORIG_MOD_MARKER, ".")


def adapt_state_dict_to_model(
    state_dict: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Map checkpoint keys onto the layout expected by ``model``.

    ``torch.compile`` wraps compiled submodules in ``OptimizedModule`` and
    inserts ``._orig_mod.`` into those parameters' state-dict keys. Checkpoints
    saved from compiled tokenizer submodules need those markers stripped when
    loading into an uncompiled model, while the reverse is needed if the target
    model is compiled.
    """
    model_keys = list(model.state_dict().keys())
    if set(state_dict.keys()) == set(model_keys):
        return state_dict

    canonical_to_model_key = {_strip_orig_mod(key): key for key in model_keys}

    adapted: dict[str, torch.Tensor] = {}
    rewritten = 0
    for key, value in state_dict.items():
        target_key = canonical_to_model_key.get(_strip_orig_mod(key), key)
        if target_key != key:
            rewritten += 1
        adapted[target_key] = value

    if rewritten:
        logger.info(
            "Adapted %d/%d tokenizer state_dict keys to match model layout "
            "(torch.compile '_orig_mod' canonicalization)",
            rewritten,
            len(state_dict),
        )
    return adapted


def _model_state_dtype(state_dict: dict[str, torch.Tensor]) -> str:
    for tensor in state_dict.values():
        if tensor.is_floating_point():
            return str(tensor.dtype).removeprefix("torch.")
    return "float32"


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported checkpoint model_dtype: {dtype_name!r}")
    return dtype_by_name[dtype_name]


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
    model_state_dict = model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "model_dtype": _model_state_dtype(model_state_dict),
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
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = adapt_state_dict_to_model(model_state_dict, model)
    model.load_state_dict(model_state_dict)
    model_dtype = _dtype_from_name(
        checkpoint.get("model_dtype", _model_state_dtype(model_state_dict))
    )
    model.to(device=device, dtype=model_dtype)
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

    model_state_dict = checkpoint["model_state_dict"]
    model_dtype = _dtype_from_name(
        checkpoint.get("model_dtype", _model_state_dtype(model_state_dict))
    )

    model = create_model(config)
    model_state_dict = adapt_state_dict_to_model(model_state_dict, model)
    model.load_state_dict(model_state_dict)
    model.to(device=device, dtype=model_dtype)
    model.eval()

    return model, config
