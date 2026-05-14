# python -m scripts.latent_action_model.train --frames_dir pokemon --num_images_in_video 5 --batch_size 2 --save_interval 100 --num_epochs 15
# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import math
import os
import time
from datetime import datetime
from typing import Callable, Optional

import torch
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from data.s3.s3_utils import S3Manager, default_s3_manager
from latent_action_model.create_model import create_action_model
from latent_action_model.model import LatentActionVQVAE
from latent_action_model.training_args import VideoTrainingConfig
from loss.loss_fns import (
    clipped_next_frame_reconstruction_loss,
    clipped_next_frame_reconstruction_residual_loss,
    next_frame_reconstruction_loss,
    next_frame_reconstruction_residual_loss,
)
from monitoring.action_code_counts import format_top_code_counts, get_top_code_counts
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.experiment_logger import ExperimentLogger, resolve_logging_backend
from monitoring.videos import (
    convert_video_to_images,
    save_comparison_images_next_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# beartype_all(conf=BeartypeConf(violation_type=UserWarning))


def upload_logs_to_s3(config: VideoTrainingConfig, s3_manager: S3Manager):
    """Upload logs to S3"""
    if config._temp_log_file and os.path.exists(config._temp_log_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_log_key = (
            f"{config.s3_logs_prefix}/{config.experiment_name}/training_{timestamp}.log"
        )

        success = s3_manager.upload_file(config._temp_log_file, s3_log_key)
        if success:
            logger.info(f"Uploaded logs to S3: {s3_log_key}")
        else:
            logging.error(f"Failed to upload logs to S3: {s3_log_key}")


def save_checkpoint(
    model: LatentActionVQVAE,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_idx: int,
    loss: float,
    config: VideoTrainingConfig,
    dataloader_state: dict,
    checkpoint_dir: str,
    s3_manager: Optional[S3Manager] = None,
    is_best=False,
    global_step: int | None = None,
):
    """Save comprehensive model checkpoint to local storage or S3"""

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config.__dict__,
        "dataloader_state": dataloader_state,
        "timestamp": datetime.now().isoformat(),
        "global_step": global_step,
    }

    if config.use_s3 and s3_manager:
        # Save to S3
        checkpoint_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        latest_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_latest.pt"

        # Upload checkpoint
        success = s3_manager.upload_pytorch_model(checkpoint, checkpoint_key)
        if success:
            logger.info(f"Checkpoint saved to S3: {checkpoint_key}")

            # Also save as latest
            s3_manager.upload_pytorch_model(checkpoint, latest_key)

            # Save best checkpoint if this is the best
            if is_best:
                best_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_best.pt"
                s3_manager.upload_pytorch_model(checkpoint, best_key)
                logger.info(f"New best checkpoint saved to S3: {best_key}")

            return checkpoint_key
        else:
            logging.error(f"Failed to save checkpoint to S3: {checkpoint_key}")
            return None
    else:
        # Save locally
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

        # Save best checkpoint if this is the best
        if is_best:
            best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model: LatentActionVQVAE,
    optimizer,
    scheduler,
    device,
    s3_manager: Optional[S3Manager] = None,
):
    """Load comprehensive model checkpoint from local storage or S3"""

    if checkpoint_path.startswith("s3://") or (
        s3_manager and not os.path.exists(checkpoint_path)
    ):
        # Load from S3
        if s3_manager is None:
            logging.error("S3Manager required for S3 checkpoint loading")
            return None

        checkpoint = s3_manager.download_pytorch_model(
            checkpoint_path, map_location=device
        )
        if checkpoint is None:
            logging.error(f"Checkpoint not found in S3: {checkpoint_path}")
            return None
    else:
        # Load locally
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except (KeyError, ValueError) as exc:
        logger.warning(
            "Skipping optimizer/scheduler state load from %s due to state mismatch: %s",
            checkpoint_path,
            exc,
        )

    epoch = checkpoint["epoch"]
    batch_idx = checkpoint.get("batch_idx", 0)
    loss = checkpoint["loss"]
    dataloader_state = checkpoint.get("dataloader_state", {})

    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    logger.info(f"Resuming from epoch {epoch}, batch {batch_idx}, loss: {loss:.6f}")

    return {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss": loss,
        "config": checkpoint["config"],
        "dataloader_state": dataloader_state,
        "global_step": checkpoint.get("global_step", 0),
    }


def build_action_loss_fn(
    config: VideoTrainingConfig,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if config.action_decoder_loss == "l2":
        return (
            next_frame_reconstruction_residual_loss
            if config.predict_action_residuals
            else next_frame_reconstruction_loss
        )
    if config.action_decoder_loss == "clipped_l2":
        clipped_loss_fn = (
            clipped_next_frame_reconstruction_residual_loss
            if config.predict_action_residuals
            else clipped_next_frame_reconstruction_loss
        )

        def action_loss(video: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
            return clipped_loss_fn(
                video,
                decoded,
                l2_clip_c=config.action_l2_clip_c,
            )

        return action_loss
    raise ValueError(f"Unknown action_decoder_loss: {config.action_decoder_loss!r}")


def run_action_model(
    model: LatentActionVQVAE, video_batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_encoded = model.encode(video_batch)
    decoded = model.decode(video_batch, action_encoded)
    action_tokens = model.get_action_sequence(action_encoded)
    return decoded, action_tokens, action_encoded


def evaluate_model(
    model: LatentActionVQVAE,
    dataloader: VideoWindowLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    global_step: int,
    config: VideoTrainingConfig,
    num_batches: int = 10,
    experiment_logger: ExperimentLogger | None = None,
    split: str = "eval",
) -> float:
    """Evaluate the action model on held-out clips and save comparison grids."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    action_tokens_accum: list[torch.Tensor] = []
    saved_next_frame_paths: list[str] = []
    saved_residual_paths: list[str] = []

    eval_dir = f"{config.save_dir}/{split}/epoch_{epoch}"
    os.makedirs(eval_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                video_batch = video_batch.to(device)
                decoded, action_tokens, _ = run_action_model(model, video_batch)
                loss = criterion(video_batch, decoded)

                total_loss += loss.item() * video_batch.size(0)
                total_samples += video_batch.size(0)
                action_tokens_accum.append(action_tokens.detach().cpu())

                if config.predict_action_residuals:
                    reconstructed_next_frames = reconstruct_predicted_frames(
                        video_batch, decoded
                    )
                    residuals = decoded
                else:
                    reconstructed_next_frames = decoded
                    residuals = None

                batch_dir = f"{eval_dir}/batch_{batch_idx}"
                next_frame_path, residual_path = save_visualizations(
                    video_batch=video_batch,
                    action_tokens=action_tokens,
                    reconstructed_next_frames=reconstructed_next_frames,
                    predicted_residuals=residuals,
                    output_dir=batch_dir,
                )
                saved_next_frame_paths.append(next_frame_path)
                if residual_path is not None:
                    saved_residual_paths.append(residual_path)

            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    metrics: dict[str, object] = {f"{split}/loss": avg_loss}
    if action_tokens_accum:
        action_tokens_flat = torch.cat(
            [tokens.reshape(-1) for tokens in action_tokens_accum], dim=0
        )
        metrics.update(
            {
                f"{split}/action_codebook_{key}": value
                for key, value in compute_codebook_usage(
                    action_tokens_flat, model.action_vocab_size
                ).items()
            }
        )
        metrics[f"{split}/top_action_codes"] = format_top_code_counts(
            get_top_code_counts(action_tokens_flat, model.action_vocab_size)
        )

    if experiment_logger:
        experiment_logger.log(metrics, step=global_step)
        experiment_logger.log_image_batches(
            key_prefix=f"{split}/next_frame_comparison",
            image_paths=saved_next_frame_paths,
            batch_size=5,
            step=global_step,
        )
        experiment_logger.log_image_batches(
            key_prefix=f"{split}/residual_comparison",
            image_paths=saved_residual_paths,
            batch_size=5,
            step=global_step,
        )

    model.train()
    return avg_loss


def reconstruct_predicted_frames(
    video_batch: torch.Tensor, decoded_residuals: torch.Tensor
) -> torch.Tensor:
    """Convert residual predictions [B, T-1, C, H, W] into next-frame predictions."""
    return torch.clamp(
        video_batch[:, :-1, :, :, :] + decoded_residuals,
        0.0,
        1.0,
    )


def _save_residual_grid(
    predicted_residual_videos,  # list[list[PIL.Image.Image]]
    target_residual_videos,  # list[list[PIL.Image.Image]]
    file_prefix: str,
) -> None:
    """Save a grid visualization of residuals to show predicted vs target movement."""
    import matplotlib.pyplot as plt
    import numpy as np

    num_samples = len(predicted_residual_videos)
    if num_samples == 0:
        return

    num_frames = len(predicted_residual_videos[0])

    # We create 2 rows per sample: one for predicted residuals, one for target residuals.
    # Shape: [2 * num_samples, num_frames]
    num_rows = num_samples * 2 if target_residual_videos is not None else num_samples

    fig, axs = plt.subplots(
        num_rows,
        num_frames,
        figsize=(num_frames * 2.5, num_rows * 2.0),
    )

    # Normalize axes array for single row/column cases
    if num_rows == 1:
        axs = np.expand_dims(axs, 0)
    if num_frames == 1:
        axs = np.expand_dims(axs, 1)

    for i in range(num_samples):
        pred_video = predicted_residual_videos[i]
        tgt_video = (
            target_residual_videos[i] if target_residual_videos is not None else None
        )

        # Base row index for this sample
        base_row = i * 2 if tgt_video is not None else i

        for j, frame in enumerate(pred_video):
            # Predicted residuals
            axs[base_row, j].imshow(frame)
            axs[base_row, j].set_title(f"Pred S{i}F{j}")
            axs[base_row, j].axis("off")

            # Target residuals (if provided)
            if tgt_video is not None and j < len(tgt_video):
                axs[base_row + 1, j].imshow(tgt_video[j])
                axs[base_row + 1, j].set_title(f"Target S{i}F{j}")
                axs[base_row + 1, j].axis("off")

    plt.tight_layout()
    os.makedirs(file_prefix, exist_ok=True)
    plt.savefig(os.path.join(file_prefix, "residual_grid.png"))
    plt.close(fig)


def save_visualizations(
    video_batch: torch.Tensor,
    action_tokens: torch.Tensor,
    reconstructed_next_frames: torch.Tensor,
    predicted_residuals: torch.Tensor | None,
    output_dir: str,
) -> tuple[str, str | None]:
    """Save next-frame and optional signed-residual comparison grids."""
    max_samples = 4
    video_batch_vis = video_batch[:max_samples]
    action_tokens_vis = action_tokens[:max_samples]
    next_frames_vis = reconstructed_next_frames[:max_samples]

    predicted_videos = convert_video_to_images(next_frames_vis)
    expected_videos = convert_video_to_images(video_batch_vis)

    save_comparison_images_next_frame(
        predicted_videos,
        action_tokens_vis.squeeze(-1).detach().cpu().numpy().tolist(),
        expected_videos,
        output_dir,
    )
    next_frame_path = os.path.join(output_dir, "next_frame_comparison_grid.png")

    residual_path: str | None = None
    if predicted_residuals is not None:
        residual_videos = convert_video_to_images(
            predicted_residuals[:max_samples],
            value_mode="signed_residual",
        )
        save_comparison_images_next_frame(
            residual_videos,
            action_tokens_vis.squeeze(-1).detach().cpu().numpy().tolist(),
            expected_videos,
            output_dir,
            file_suffix="residual_comparison_grid.png",
            predicted_label="Predicted Residual",
        )
        residual_path = os.path.join(output_dir, "residual_comparison_grid.png")

    return next_frame_path, residual_path


def train_epoch(
    model: LatentActionVQVAE,
    train_dataloader: VideoWindowLoader,
    eval_dataloader: Optional[VideoWindowLoader],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    config: VideoTrainingConfig,
    experiment_logger: ExperimentLogger | None,
    global_step: int,
    best_loss: float,
    s3_manager: Optional[S3Manager] = None,
    start_batch: int = 0,
) -> tuple[float, int, float]:
    """Train for one epoch with comprehensive logging"""
    total_optimizer_step_loss = 0.0
    num_batches = len(train_dataloader)
    accumulation_steps = config.gradient_accumulation_steps

    # Configure the resumable dataloader: set the per-epoch shuffle and skip
    # ahead to ``start_batch`` at the sampler level, so workers never load
    # (and discard) the skipped batches.
    train_dataloader.resumable_loader.set_epoch(epoch)
    train_dataloader.resumable_loader.set_start_batch(start_batch)

    epoch_start_time = time.time()
    loss_acc: list[float] = []
    logged_action_tokens: list[torch.Tensor] = []
    optimizer.zero_grad()

    # ``enumerate(..., start=start_batch)`` keeps ``batch_idx`` aligned with
    # the absolute position in the (full) epoch, so gradient accumulation,
    # logging, and end-of-epoch detection all match a fresh-start run.
    for batch_idx, video_batch in enumerate(train_dataloader, start=start_batch):
        batch_start_time = time.time()

        video_batch = video_batch.to(device)

        decoded, action_tokens, _ = run_action_model(model, video_batch)
        loss = criterion(video_batch, decoded)

        remaining_batches = num_batches - batch_idx
        current_window_size = min(accumulation_steps, remaining_batches)
        (loss / current_window_size).backward()
        loss_acc.append(loss.item())
        logged_action_tokens.append(action_tokens.detach().cpu())

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clipping
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = sum(loss_acc) / len(loss_acc)
            total_optimizer_step_loss += avg_loss
            action_token_window = (
                torch.cat([tokens.reshape(-1) for tokens in logged_action_tokens], dim=0)
                if logged_action_tokens
                else torch.empty(0, dtype=torch.long)
            )
            codebook_usage = compute_codebook_usage(
                action_token_window, model.action_vocab_size
            )
            top_code_counts = get_top_code_counts(
                action_token_window, model.action_vocab_size
            )

            batch_time = time.time() - batch_start_time
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            if experiment_logger:
                metrics: dict[str, object] = {
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "train/top_action_codes": format_top_code_counts(top_code_counts),
                }
                metrics.update(
                    {
                        f"train/action_codebook_{key}": value
                        for key, value in codebook_usage.items()
                    }
                )
                experiment_logger.log(metrics, step=global_step)

            if batch_idx % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {avg_loss:.6f}, LR: {current_lr:.2e}, "
                    f"Codebook usage: {codebook_usage['usage_fraction']:.3f}, "
                    f"Top codes: {format_top_code_counts(top_code_counts)}, "
                    f"Time: {batch_time:.2f}s"
                )

            if global_step % config.save_interval == 0:
                if config.predict_action_residuals:
                    reconstructed_next_frames = reconstruct_predicted_frames(
                        video_batch, decoded
                    )
                    residuals = decoded
                else:
                    reconstructed_next_frames = decoded
                    residuals = None

                train_batch_dir = (
                    f"{config.save_dir}/train/epoch_{epoch}/batch_{batch_idx}"
                )
                next_frame_path, residual_path = save_visualizations(
                    video_batch=video_batch,
                    action_tokens=action_tokens,
                    reconstructed_next_frames=reconstructed_next_frames,
                    predicted_residuals=residuals,
                    output_dir=train_batch_dir,
                )
                if experiment_logger:
                    experiment_logger.log_image(
                        "train/next_frame_comparison",
                        next_frame_path,
                        step=global_step,
                    )
                    if residual_path is not None:
                        experiment_logger.log_image(
                            "train/residual_comparison",
                            residual_path,
                            step=global_step,
                        )

                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    batch_idx,
                    avg_loss,
                    config,
                    train_dataloader.get_state(),
                    config.checkpoint_dir,
                    s3_manager,
                    is_best,
                    global_step=global_step,
                )

            if (
                eval_dataloader is not None
                and global_step % config.eval_interval == 0
            ):
                eval_loss = evaluate_model(
                    model,
                    eval_dataloader,
                    criterion,
                    device,
                    epoch,
                    global_step,
                    config,
                    experiment_logger=experiment_logger,
                )
                logger.info(f"Evaluation loss at step {global_step}: {eval_loss:.6f}")

            loss_acc, logged_action_tokens = [], []

    num_batches_processed = max(num_batches - start_batch, 0)
    num_optimizer_steps = math.ceil(num_batches_processed / accumulation_steps)
    avg_loss = total_optimizer_step_loss / max(num_optimizer_steps, 1)
    epoch_time = time.time() - epoch_start_time

    if experiment_logger:
        experiment_logger.log(
            {
                "train/epoch_loss": avg_loss,
                "train/epoch_time": epoch_time,
                "train/epoch": epoch,
            },
            step=global_step,
        )

    logger.info(
        f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s"
    )
    return avg_loss, global_step, best_loss


def main(config: VideoTrainingConfig):
    """Main training function with resumable training support and S3 integration"""

    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"pokemon_vqvae_{timestamp}"

    logging_backend = resolve_logging_backend(
        logging_backend=config.logging_backend,
        use_wandb=config.use_wandb,
    )
    experiment_logger = ExperimentLogger(
        backend=logging_backend,
        run_name=config.experiment_name,
        config_summary=config.__dict__,
        group="latent-action-model",
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_tags=config.wandb_tags or [],
        wandb_notes=config.wandb_notes or "",
        tensorboard_dir=config.tensorboard_dir,
    )

    logger.info(
        f"Starting Pokemon VQVAE training - Experiment: {config.experiment_name}"
    )
    logger.info(f"Using S3: {config.use_s3}")
    logger.info(f"Logging backend: {logging_backend}")
    if experiment_logger and experiment_logger.tensorboard_log_dir is not None:
        logger.info(f"TensorBoard log dir: {experiment_logger.tensorboard_log_dir}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Create device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Create data loader
    logger.info("Creating data loader...")
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")

    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )

    train_dataset, test_dataset = build_datasets(config, local_cache)

    logger.info("Creating data loaders...")
    train_dataloader = VideoWindowLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
    )
    eval_dataloader = VideoWindowLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
    )

    # Print dataset info
    train_info = train_dataloader.get_dataset_info()
    eval_info = eval_dataloader.get_dataset_info()
    logger.info("Dataset Info (train):")
    for key, value in train_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("Dataset Info (eval):")
    for key, value in eval_info.items():
        logger.info(f"  {key}: {value}")

    if experiment_logger:
        experiment_logger.log(
            {f"dataset/{key}": value for key, value in train_info.items()},
            commit=False,
        )
        experiment_logger.log(
            {f"eval_dataset/{key}": value for key, value in eval_info.items()},
            commit=False,
        )
        experiment_logger.log(
            {
                "config/effective_batch_size": config.batch_size
                * config.gradient_accumulation_steps,
                "config/gradient_accumulation_steps": config.gradient_accumulation_steps,
            },
            commit=False,
        )

    # Create model
    logger.info(f"Creating model on device {device}...")
    model = create_action_model(config)
    model.to(device)

    # Watch model with wandb
    if experiment_logger:
        experiment_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    logger.info(f"Optimizer created with learning rate: {config.learning_rate}")

    steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=config.min_learning_rate,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=config.min_learning_rate,
        )
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total optimizer steps: {total_steps}")

    criterion = build_action_loss_fn(config)
    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_loss = float("inf")

    if config.resume_from:
        checkpoint_info = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, device, s3_manager
        )
        if checkpoint_info:
            start_epoch = checkpoint_info["epoch"]
            start_batch = checkpoint_info.get("batch_idx", 0)
            global_step = checkpoint_info.get("global_step", 0)
            best_loss = checkpoint_info["loss"]

            # Restore dataloader state
            if "dataloader_state" in checkpoint_info:
                dataloader_state = checkpoint_info["dataloader_state"]
                train_dataloader.resumable_loader = (
                    train_dataloader.create_resumable_loader(start_epoch, start_batch)
                )
            if start_batch >= len(train_dataloader):
                start_epoch += 1
                start_batch = 0

    # Training loop
    logger.info("Starting training loop...")
    epoch = start_epoch

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss, global_step, best_loss = train_epoch(
                model,
                train_dataloader,
                eval_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                config,
                experiment_logger,
                global_step,
                best_loss,
                s3_manager,
                epoch_start_batch,
            )

            is_best = avg_loss <= best_loss
            if is_best:
                best_loss = avg_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                len(train_dataloader),
                avg_loss,
                config,
                train_dataloader.get_state(),
                config.checkpoint_dir,
                s3_manager,
                is_best,
                global_step=global_step,
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        dataloader_state = train_dataloader.get_state()
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_dataloader.resumable_loader.current_batch,
            best_loss,
            config,
            dataloader_state,
            config.checkpoint_dir,
            s3_manager,
            False,
            global_step=global_step,
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        # Finish wandb run
        if experiment_logger:
            experiment_logger.finish()

        # Cleanup temporary directories
        if config._temp_log_file and os.path.exists(config._temp_log_file):
            os.unlink(config._temp_log_file)
        if config._temp_tensorboard_dir and os.path.exists(
            config._temp_tensorboard_dir
        ):
            import shutil

            shutil.rmtree(config._temp_tensorboard_dir)

    logger.info("Training completed!")
    logger.info(f"Best loss achieved: {best_loss:.6f}")

    if config.use_s3 and s3_manager:
        logger.info(
            f"Checkpoints and logs saved to S3 bucket: {s3_manager.bucket_name}"
        )
    else:
        logger.info(
            f"Tensorboard logs saved to: {os.path.join(config.tensorboard_dir, config.experiment_name or 'default')}"
        )

    if logging_backend == "wandb":
        logger.info(f"Training metrics logged to Wandb project: {config.wandb_project}")


if __name__ == "__main__":
    config = tyro.cli(VideoTrainingConfig)

    logger.info(f"Starting training... config: {config.__dict__}")
    main(config)
    logger.info("Training completed!")
