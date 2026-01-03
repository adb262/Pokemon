"""Wandb media logging utilities."""

import logging
from typing import Sequence

import wandb
from wandb.wandb_run import Run

from monitoring.videos.image_stack import stack_image_paths_vertically

logger = logging.getLogger(__name__)


def log_image_batches(
    run: Run,
    *,
    key_prefix: str,
    image_paths: Sequence[str],
    batch_size: int = 5,
    step: int | None = None,
) -> None:
    """
    Log images to wandb in batches, stacking each batch vertically.

    Args:
        run: The wandb Run instance to log to.
        key_prefix: Prefix for the logged keys (e.g., "eval/comparison").
            Images will be logged as "{key_prefix}_0", "{key_prefix}_1", etc.
        image_paths: Sequence of paths to comparison images to log.
        batch_size: Number of images to stack per batch.
        step: Optional global step for wandb logging.

    Example:
        >>> log_image_batches(
        ...     wandb_run,
        ...     key_prefix="eval/comparison",
        ...     image_paths=saved_paths,
        ...     batch_size=5,
        ...     step=global_step,
        ... )
        # Logs: eval/comparison_0, eval/comparison_1, ...
    """
    if not image_paths:
        logger.debug("No images to log")
        return

    log_dict: dict[str, wandb.Image] = {}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_idx = i // batch_size

        try:
            stacked_image = stack_image_paths_vertically(batch_paths)
            key = f"{key_prefix}_{batch_idx}"
            log_dict[key] = wandb.Image(stacked_image)
            logger.debug(f"Prepared batch {batch_idx} with {len(batch_paths)} images")
        except Exception as e:
            logger.warning(f"Failed to stack batch {batch_idx}: {e}")
            continue

    if log_dict:
        if step is not None:
            run.log(log_dict, step=step)
        else:
            run.log(log_dict)
        logger.info(f"Logged {len(log_dict)} image batches to wandb")

