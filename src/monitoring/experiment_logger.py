"""Experiment logging backends for training scripts.

Provides a single :class:`ExperimentLogger` interface that fans out to either
Weights & Biases, TensorBoard, or a no-op backend. Training scripts construct
one instance and call ``log`` / ``log_image`` / ``log_image_batches`` without
needing to know which backend is active.
"""

import logging
import math
import os
from typing import Literal

import torch
import wandb
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from monitoring.setup_wandb import setup_wandb
from monitoring.videos.image_stack import stack_image_paths_vertically
from monitoring.wandb_media import log_image_batches as _log_image_batches_to_wandb

logger = logging.getLogger(__name__)

LoggingBackend = Literal["wandb", "tensorboard", "none"]


def resolve_logging_backend(
    *,
    logging_backend: LoggingBackend,
    use_wandb: bool,
) -> LoggingBackend:
    """Resolve the effective backend, falling back to ``"none"`` when wandb is disabled."""
    if logging_backend != "wandb":
        return logging_backend
    if not use_wandb:
        return "none"
    return "wandb"


def _format_config_text(config_dict: dict[str, object]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(config_dict.items()))


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    image_rgb = image.convert("RGB")
    image_bytes = torch.ByteTensor(torch.ByteStorage.from_buffer(image_rgb.tobytes()))
    return image_bytes.view(image_rgb.height, image_rgb.width, 3).permute(2, 0, 1)


def _load_image_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image.load()
    return _image_to_tensor(image)


class ExperimentLogger:
    """Unified W&B / TensorBoard / no-op logger for training experiments."""

    def __init__(
        self,
        *,
        backend: LoggingBackend,
        run_name: str,
        config_summary: dict[str, object],
        group: str,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
        tensorboard_dir: str | None = None,
    ) -> None:
        self.backend: LoggingBackend = backend
        self.tensorboard_log_dir: str | None = None
        self.wandb_run = None
        self.tensorboard_writer: SummaryWriter | None = None

        if backend == "wandb":
            if wandb_project is None:
                raise ValueError("wandb_project is required when backend='wandb'")
            self.wandb_run = setup_wandb(
                project=wandb_project,
                group=group,
                entity=wandb_entity,
                name=run_name,
                tags=wandb_tags or [],
                notes=wandb_notes or "",
                config=config_summary,
            )
        elif backend == "tensorboard":
            if tensorboard_dir is None:
                raise ValueError("tensorboard_dir is required when backend='tensorboard'")
            self.tensorboard_log_dir = os.path.join(tensorboard_dir, run_name)
            os.makedirs(self.tensorboard_log_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
            writer = self._require_tensorboard_writer()
            writer.add_text(
                "config",
                _format_config_text(config_summary),
                global_step=0,
            )
            writer.flush()

    def __bool__(self) -> bool:
        return self.backend != "none"

    def _require_wandb_run(self):
        if self.wandb_run is None:
            raise RuntimeError("wandb run is not initialized")
        return self.wandb_run

    def _require_tensorboard_writer(self) -> SummaryWriter:
        if self.tensorboard_writer is None:
            raise RuntimeError("TensorBoard writer is not initialized")
        return self.tensorboard_writer

    def log(
        self,
        metrics: dict[str, object],
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        if self.backend == "none":
            return

        if self.backend == "wandb":
            run = self._require_wandb_run()
            if step is None:
                run.log(metrics, commit=commit)
            else:
                run.log(metrics, step=step, commit=commit)
            return

        writer = self._require_tensorboard_writer()
        for key, value in metrics.items():
            if isinstance(value, (bool, int, float)):
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    writer.add_scalar(
                        key,
                        numeric_value,
                        global_step=step,
                    )
                else:
                    logger.warning(f"Skipping non-finite TensorBoard metric {key}={value}")
            else:
                writer.add_text(
                    key,
                    str(value),
                    global_step=step,
                )
        writer.flush()

    def log_image(self, key: str, image_path: str, *, step: int | None = None) -> None:
        if self.backend == "none":
            return

        if self.backend == "wandb":
            run = self._require_wandb_run()
            run.log({key: wandb.Image(image_path)}, step=step)
            return

        writer = self._require_tensorboard_writer()
        writer.add_image(
            key,
            _load_image_tensor(image_path),
            global_step=step,
        )
        writer.flush()

    def log_image_batches(
        self,
        *,
        key_prefix: str,
        image_paths: list[str],
        batch_size: int = 5,
        step: int | None = None,
    ) -> None:
        if self.backend == "none" or not image_paths:
            return

        if self.backend == "wandb":
            run = self._require_wandb_run()
            _log_image_batches_to_wandb(
                run,
                key_prefix=key_prefix,
                image_paths=image_paths,
                batch_size=batch_size,
                step=step,
            )
            return

        writer = self._require_tensorboard_writer()
        for start_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start_idx : start_idx + batch_size]
            batch_idx = start_idx // batch_size
            stacked_image = stack_image_paths_vertically(batch_paths)
            writer.add_image(
                f"{key_prefix}_{batch_idx}",
                _image_to_tensor(stacked_image),
                global_step=step,
            )
        writer.flush()

    def watch(
        self,
        model: torch.nn.Module,
        *,
        log: Literal["gradients", "parameters", "all"],
        log_freq: int,
    ) -> None:
        if self.backend == "wandb":
            run = self._require_wandb_run()
            run.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        if self.backend == "wandb":
            run = self._require_wandb_run()
            run.finish()
        elif self.backend == "tensorboard":
            writer = self._require_tensorboard_writer()
            writer.close()
