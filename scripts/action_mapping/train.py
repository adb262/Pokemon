#!/usr/bin/env python3
"""Train an action mapping model on generated two-player Pong data.

Example:
    python -m scripts.action_mapping.train \
      --data-dir data/two_player_pong \
      --tokenizer-checkpoint-path tokenizer.pt \
      --dynamics-model-checkpoint-path dynamics.pt \
      --action-model-checkpoint-path action_model.pt \
      --logging-backend tensorboard
"""

import logging
import math
import os
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from action_mapping.model import ActionMappingModel
from dynamics_model.checkpoints import adapt_state_dict_to_model
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.model import DynamicsModel
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from latent_action_model.model import LatentActionVQVAE
from monitoring.experiment_logger import ExperimentLogger, resolve_logging_backend
from scripts.data.generate_two_player_pong import (
    Args as PongDataArgs,
    build_metadata,
    collect_split,
    prepare_output_dir,
    save_preview,
    validate_args,
    write_metadata,
    pong_v3,
)
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ActionMappingTrainingConfig:
    data_dir: str = "data/two_player_pong"
    generate_dataset: bool = False
    overwrite_dataset: bool = False
    num_windows_train: int = 10_000
    num_windows_val: int = 1_000
    num_windows_test: int = 1_000
    generated_window_size: int = 160
    windows_per_file: int = 1024
    generated_frame_size: int = 84
    pong_policy: Literal["random", "sticky_random", "tracking"] = "random"

    tokenizer_checkpoint_path: str = ""
    dynamics_model_checkpoint_path: str = ""
    action_model_checkpoint_path: str = ""

    image_size: int = 84
    max_sequence_length: int = 5
    window_stride: int = 1
    num_input_actions: int = 9
    d_model: int | None = None
    num_heads: int = 8
    num_layers: int = 4

    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    gradient_clipping: float | None = 1.0
    warmup_steps: int = 100
    num_epochs: int = 5
    eval_batches: int = 50
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "action_mapping_checkpoints"
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    use_bf16: bool = True

    use_wandb: bool = True
    logging_backend: Literal["wandb", "tensorboard", "none"] = "tensorboard"
    tensorboard_dir: str = "action_mapping_runs"
    wandb_project: str = "pokemon-action-mapping"
    wandb_entity: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None
    experiment_name: str | None = None


class TwoPlayerPongWindowDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: str,
        split: str,
        sequence_length: int,
        image_size: int,
        stride: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        self.split_dir = Path(data_dir) / split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.stride = stride
        self.index: list[tuple[Path, int, int]] = []
        self._shard_cache: dict[Path, dict[str, np.ndarray]] = {}

        shard_paths = sorted(self.split_dir.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz files found in {self.split_dir}")

        for shard_path in shard_paths:
            with np.load(shard_path) as shard:
                frames_shape = shard["frames"].shape
            if len(frames_shape) != 5:
                raise ValueError(
                    f"Expected frames in {shard_path} to have shape NTHWC, "
                    f"got {frames_shape}"
                )
            num_windows = frames_shape[0]
            source_length = frames_shape[1]
            if source_length < sequence_length:
                continue
            for window_idx in range(num_windows):
                for start in range(0, source_length - sequence_length + 1, stride):
                    self.index.append((shard_path, window_idx, start))

        if not self.index:
            raise ValueError(
                f"No rolling windows of length {sequence_length} available in {self.split_dir}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_path: Path) -> dict[str, np.ndarray]:
        cached = self._shard_cache.get(shard_path)
        if cached is not None:
            return cached
        with np.load(shard_path) as shard:
            cached = {
                "frames": shard["frames"],
                "dual_actions": shard["dual_actions"],
            }
        self._shard_cache = {shard_path: cached}
        return cached

    def __getitem__(self, item_idx: int) -> dict[str, torch.Tensor]:
        shard_path, window_idx, start = self.index[item_idx]
        shard = self._load_shard(shard_path)
        end = start + self.sequence_length

        frames = shard["frames"][window_idx, start:end]
        dual_actions = shard["dual_actions"][window_idx, start : end - 1]
        joint_actions = dual_actions[:, 0] * 3 + dual_actions[:, 1]

        video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        if video.shape[-2:] != (self.image_size, self.image_size):
            video = F.interpolate(
                video,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "video": video,
            "actions": torch.from_numpy(joint_actions.copy()).long(),
        }

    def get_dataset_info(self) -> dict[str, int | str]:
        return {
            "split_dir": str(self.split_dir),
            "num_samples": len(self),
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "image_size": self.image_size,
        }


def generate_pong_dataset(config: ActionMappingTrainingConfig) -> None:
    args = PongDataArgs(
        output_dir=config.data_dir,
        num_windows_train=config.num_windows_train,
        num_windows_val=config.num_windows_val,
        num_windows_test=config.num_windows_test,
        window_size=max(config.generated_window_size, config.max_sequence_length),
        windows_per_file=config.windows_per_file,
        frame_size=config.generated_frame_size,
        seed=config.seed,
        policy=config.pong_policy,
        overwrite=config.overwrite_dataset,
    )
    validate_args(args)
    prepare_output_dir(args)

    rng = np.random.default_rng(args.seed)
    env = pong_v3.parallel_env(num_players=2, render_mode=None)
    split_targets = {
        "train": args.num_windows_train,
        "val": args.num_windows_val,
        "test": args.num_windows_test,
    }
    split_metadata: dict[str, dict[str, object]] = {}
    episode_id = 0
    try:
        for split_name, target_windows in split_targets.items():
            metadata, episode_id = collect_split(
                env,
                split_name,
                target_windows,
                args,
                rng,
                episode_id,
            )
            split_metadata[split_name] = metadata
    finally:
        env.close()

    preview_path = save_preview(args.output_dir, args)
    metadata = build_metadata(args, split_metadata, preview_path)
    metadata_path = write_metadata(args.output_dir, metadata)
    logger.info("Generated Pong dataset metadata at %s", metadata_path)


def build_dataloaders(
    config: ActionMappingTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TwoPlayerPongWindowDataset(
        data_dir=config.data_dir,
        split="train",
        sequence_length=config.max_sequence_length,
        image_size=config.image_size,
        stride=config.window_stride,
    )
    eval_split = "val" if (Path(config.data_dir) / "val").exists() else "test"
    eval_dataset = TwoPlayerPongWindowDataset(
        data_dir=config.data_dir,
        split=eval_split,
        sequence_length=config.max_sequence_length,
        image_size=config.image_size,
        stride=config.window_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info("Train dataset: %s", train_dataset.get_dataset_info())
    logger.info("Eval dataset: %s", eval_dataset.get_dataset_info())
    return train_loader, eval_loader


def dynamics_config_from_checkpoint(checkpoint_path: str) -> DynamicsModelTrainingConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = DynamicsModelTrainingConfig()
    config_keys = {field.name for field in fields(DynamicsModelTrainingConfig)}
    for key, value in checkpoint.get("config", {}).items():
        if key in config_keys:
            setattr(config, key, value)
    return config


def load_action_model_checkpoint(
    action_model: LatentActionVQVAE,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = adapt_state_dict_to_model(checkpoint["model_state_dict"], action_model)
    action_model.load_state_dict(state_dict)


def load_frozen_models(
    config: ActionMappingTrainingConfig,
    device: torch.device,
) -> tuple[VideoTokenizer, LatentActionVQVAE, DynamicsModel]:
    if not config.tokenizer_checkpoint_path:
        raise ValueError("tokenizer_checkpoint_path is required")
    if not config.dynamics_model_checkpoint_path:
        raise ValueError("dynamics_model_checkpoint_path is required")
    if not config.action_model_checkpoint_path:
        raise ValueError("action_model_checkpoint_path is required")

    tokenizer, _tokenizer_config = load_model_from_checkpoint(
        config.tokenizer_checkpoint_path,
        device,
    )

    dynamics_config = dynamics_config_from_checkpoint(config.dynamics_model_checkpoint_path)
    dynamics_config.tokenizer_checkpoint_path = config.tokenizer_checkpoint_path
    dynamics_config.action_model_checkpoint_path = None

    action_model = create_action_model_from_dynamics_config(dynamics_config).to(device)
    load_action_model_checkpoint(action_model, config.action_model_checkpoint_path, device)

    dynamics_model = create_dynamics_model(dynamics_config, tokenizer, action_model).to(device)
    checkpoint = torch.load(config.dynamics_model_checkpoint_path, map_location=device)
    state_dict = adapt_state_dict_to_model(
        checkpoint["model_state_dict"],
        dynamics_model,
    )
    dynamics_model.load_state_dict(state_dict)
    load_action_model_checkpoint(action_model, config.action_model_checkpoint_path, device)

    for module in (tokenizer, action_model, dynamics_model):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    logger.info("Loaded frozen tokenizer, action model, and dynamics model")
    return tokenizer, action_model, dynamics_model


@torch.no_grad()
def compute_video_token_latents(
    dynamics_model: DynamicsModel,
    video: torch.Tensor,
) -> torch.Tensor:
    quantized = dynamics_model.tokenizer.encode(video)
    codes = dynamics_model.tokenizer.quantized_value_to_codes(quantized).long()
    return dynamics_model.tokenizer_embedding(codes)


@torch.no_grad()
def compute_target_actions(
    action_model: LatentActionVQVAE,
    video: torch.Tensor,
) -> torch.Tensor:
    quantized = action_model.encode(video)
    target_actions = action_model.get_action_sequence(quantized).long()
    if target_actions.dim() != 2:
        raise ValueError(
            "Expected action model to return target actions with shape (B, T-1), "
            f"got {tuple(target_actions.shape)}"
        )
    return target_actions


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    flat_predictions = predictions.reshape(-1)
    flat_targets = targets.reshape(-1)
    accuracy = (flat_predictions == flat_targets).float().mean().item()
    top_k = min(3, vocab_size)
    topk_predictions = logits.topk(top_k, dim=-1).indices
    topk_accuracy = (
        topk_predictions.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    )
    target_counts = torch.bincount(flat_targets, minlength=vocab_size).float()
    pred_counts = torch.bincount(flat_predictions, minlength=vocab_size).float()
    target_probs = target_counts / target_counts.sum().clamp_min(1.0)
    pred_probs = pred_counts / pred_counts.sum().clamp_min(1.0)
    target_entropy = -(target_probs * torch.log(target_probs.clamp_min(1e-12))).sum()
    pred_entropy = -(pred_probs * torch.log(pred_probs.clamp_min(1e-12))).sum()
    return {
        "accuracy": accuracy,
        "top3_accuracy": topk_accuracy,
        "target_unique": float((target_counts > 0).sum().item()),
        "pred_unique": float((pred_counts > 0).sum().item()),
        "target_entropy": float(target_entropy.item()),
        "pred_entropy": float(pred_entropy.item()),
    }


@torch.no_grad()
def evaluate(
    model: ActionMappingModel,
    action_model: LatentActionVQVAE,
    dynamics_model: DynamicsModel,
    dataloader: DataLoader,
    device: torch.device,
    config: ActionMappingTrainingConfig,
    global_step: int,
    experiment_logger: ExperimentLogger | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    logits_accum: list[torch.Tensor] = []
    targets_accum: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(dataloader):
        if config.eval_batches > 0 and batch_idx >= config.eval_batches:
            break
        video = batch["video"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        video_token_latents = compute_video_token_latents(dynamics_model, video)
        target_actions = compute_target_actions(action_model, video)
        logits = model(video_token_latents, actions)
        loss = model.compute_loss(logits, target_actions)
        num_tokens = target_actions.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        logits_accum.append(logits.detach().cpu())
        targets_accum.append(target_actions.detach().cpu())

    avg_loss = total_loss / max(total_tokens, 1)
    metrics: dict[str, object] = {"eval/loss": avg_loss}
    if logits_accum:
        eval_metrics = compute_metrics(
            torch.cat(logits_accum, dim=0),
            torch.cat(targets_accum, dim=0),
            model.num_output_actions,
        )
        metrics.update({f"eval/{key}": value for key, value in eval_metrics.items()})

    pred_unique = metrics.get("eval/pred_unique", 0.0)
    logger.info(
        "Eval step %s: loss=%.6f accuracy=%.4f top3=%.4f pred_unique=%s/%s",
        global_step,
        avg_loss,
        metrics.get("eval/accuracy", float("nan")),
        metrics.get("eval/top3_accuracy", float("nan")),
        int(pred_unique) if isinstance(pred_unique, (int, float)) else 0,
        model.num_output_actions,
    )
    if experiment_logger:
        experiment_logger.log(metrics, step=global_step)
    model.train()
    return avg_loss


def save_checkpoint(
    model: ActionMappingModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    config: ActionMappingTrainingConfig,
    epoch: int,
    global_step: int,
    loss: float,
    best_loss: float,
    is_best: bool,
) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.__dict__,
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "best_loss": best_loss,
    }
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_epoch{epoch}_step{global_step}.pt",
    )
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, "checkpoint_latest.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, "checkpoint_best.pt"))
    logger.info("Saved checkpoint: %s", checkpoint_path)


def main(config: ActionMappingTrainingConfig) -> None:
    if config.experiment_name is None:
        config.experiment_name = f"action_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.log_interval <= 0:
        raise ValueError("log_interval must be greater than 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if config.save_interval <= 0:
        raise ValueError("save_interval must be greater than 0")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if config.generate_dataset:
        generate_pong_dataset(config)

    device = torch.device(config.device)
    logging_backend = resolve_logging_backend(
        logging_backend=config.logging_backend,
        use_wandb=config.use_wandb,
    )
    experiment_logger = ExperimentLogger(
        backend=logging_backend,
        run_name=config.experiment_name,
        config_summary=config.__dict__,
        group="action-mapping-training",
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_tags=config.wandb_tags,
        wandb_notes=config.wandb_notes,
        tensorboard_dir=config.tensorboard_dir,
    )
    if experiment_logger and experiment_logger.tensorboard_log_dir is not None:
        logger.info("TensorBoard log dir: %s", experiment_logger.tensorboard_log_dir)

    train_loader, eval_loader = build_dataloaders(config)
    _tokenizer, action_model, dynamics_model = load_frozen_models(config, device)
    if config.max_sequence_length != dynamics_model.num_images_in_video:
        raise ValueError(
            "max_sequence_length must match the dynamics checkpoint window: "
            f"{config.max_sequence_length} vs {dynamics_model.num_images_in_video}"
        )
    if config.image_size != action_model.image_height:
        raise ValueError(
            "image_size must match the action/dynamics checkpoint image size: "
            f"{config.image_size} vs {action_model.image_height}"
        )

    mapping_d_model = config.d_model
    if mapping_d_model is None:
        mapping_d_model = dynamics_model.d_model

    if mapping_d_model != dynamics_model.d_model:
        raise ValueError(
            f"Action mapping d_model ({mapping_d_model}) must match dynamics "
            f"embedding dim ({dynamics_model.d_model})"
        )

    model = ActionMappingModel(
        num_input_actions=config.num_input_actions,
        num_output_actions=action_model.action_vocab_size,
        max_sequence_length=config.max_sequence_length,
        d_model=mapping_d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    logger.info(
        "Action mapping params: %s",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_steps = max(config.num_epochs * steps_per_epoch, 1)
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 1))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=config.min_learning_rate,
            ),
        ],
        milestones=[warmup_steps],
    )

    if experiment_logger:
        experiment_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    use_amp = config.use_bf16 and device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
    global_step = 0
    best_loss = float("inf")

    try:
        for epoch in range(config.num_epochs):
            model.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_tokens = 0
            step_loss_accum = 0.0
            step_tokens_accum = 0
            step_logits_accum: list[torch.Tensor] = []
            step_targets_accum: list[torch.Tensor] = []
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader):
                video = batch["video"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)

                with torch.no_grad():
                    video_token_latents = compute_video_token_latents(dynamics_model, video)
                    target_actions = compute_target_actions(action_model, video)

                with amp_ctx:
                    logits = model(video_token_latents, actions)
                    loss = model.compute_loss(logits, target_actions)

                accumulation_boundary = (
                    (batch_idx + 1) % config.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(train_loader)
                )
                loss_for_backward = loss / config.gradient_accumulation_steps
                loss_for_backward.backward()

                num_tokens = target_actions.numel()
                epoch_loss += loss.item() * num_tokens
                epoch_tokens += num_tokens
                step_loss_accum += loss.item() * num_tokens
                step_tokens_accum += num_tokens
                step_logits_accum.append(logits.detach().cpu())
                step_targets_accum.append(target_actions.detach().cpu())

                if accumulation_boundary:
                    if config.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.gradient_clipping,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    train_loss = step_loss_accum / max(step_tokens_accum, 1)
                    train_metrics = compute_metrics(
                        torch.cat(step_logits_accum, dim=0),
                        torch.cat(step_targets_accum, dim=0),
                        model.num_output_actions,
                    )
                    log_metrics = {
                        "train/loss": train_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/batch": batch_idx,
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                    }
                    if experiment_logger:
                        experiment_logger.log(log_metrics, step=global_step)

                    if global_step % config.log_interval == 0:
                        logger.info(
                            "Step %s epoch %s batch %s/%s loss=%.6f "
                            "accuracy=%.4f top3=%.4f lr=%.2e",
                            global_step,
                            epoch,
                            batch_idx,
                            len(train_loader),
                            train_loss,
                            train_metrics["accuracy"],
                            train_metrics["top3_accuracy"],
                            scheduler.get_last_lr()[0],
                        )

                    step_loss_accum = 0.0
                    step_tokens_accum = 0
                    step_logits_accum = []
                    step_targets_accum = []

                    if global_step % config.eval_interval == 0:
                        eval_loss = evaluate(
                            model,
                            action_model,
                            dynamics_model,
                            eval_loader,
                            device,
                            config,
                            global_step,
                            experiment_logger,
                        )
                        is_best = eval_loss < best_loss
                        if is_best:
                            best_loss = eval_loss
                        if global_step % config.save_interval == 0 or is_best:
                            save_checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                config,
                                epoch,
                                global_step,
                                eval_loss,
                                best_loss,
                                is_best,
                            )
                    elif global_step % config.save_interval == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            config,
                            epoch,
                            global_step,
                            train_loss,
                            best_loss,
                            False,
                        )

            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch %s complete: loss=%.6f time=%.2fs",
                epoch,
                avg_epoch_loss,
                epoch_time,
            )
            if experiment_logger:
                experiment_logger.log(
                    {
                        "train/epoch_loss": avg_epoch_loss,
                        "train/epoch_time": epoch_time,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        final_eval_loss = evaluate(
            model,
            action_model,
            dynamics_model,
            eval_loader,
            device,
            config,
            global_step,
            experiment_logger,
        )
        final_is_best = final_eval_loss < best_loss
        if final_is_best:
            best_loss = final_eval_loss
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            config.num_epochs - 1,
            global_step,
            final_eval_loss,
            best_loss,
            final_is_best,
        )
    finally:
        if experiment_logger:
            experiment_logger.finish()

    logger.info("Training complete. Best eval loss: %.6f", best_loss)


if __name__ == "__main__":
    main(tyro.cli(ActionMappingTrainingConfig))
