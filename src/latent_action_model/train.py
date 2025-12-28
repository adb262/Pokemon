# python -m src.latent_action_model.train --use_s3 true --frames_dir pokemon --num_images_in_video 5 --batch_size 2
# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from data.data_loaders.pokemon_open_world_loader import PokemonOpenWorldLoader
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)
from data.s3.s3_utils import S3Manager, default_s3_manager
from latent_action_model.latent_action_vq_vae import LatentActionVQVAE
from latent_action_model.training_args import VideoTrainingConfig
from loss.loss_fns import next_frame_reconstruction_loss
from video_tokenization.eval import (
    convert_video_to_images,
    save_comparison_images_next_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Add the parent dir
# ectory to the path so we can import from data_collection
sys.path.append(str(Path(__file__).parent.parent))


# beartype_all(conf=BeartypeConf(violation_type=UserWarning))
# AI BS generation here... use a real config class for god sake


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


def create_model(config: VideoTrainingConfig):
    """Create and initialize the VQVAE model"""
    model = LatentActionVQVAE(
        channels=3,
        image_height=config.image_size,
        image_width=config.image_size,
        patch_height=config.patch_size,
        patch_width=config.patch_size,
        num_images_in_video=config.num_images_in_video,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.latent_dim,
        use_temporal_transformer=True,
        use_spatial_transformer=True,
    )

    return model


def save_checkpoint(
    model: LatentActionVQVAE,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    epoch: int,
    batch_idx: int,
    loss: float,
    config: VideoTrainingConfig,
    dataloader_state: dict,
    checkpoint_dir: str,
    s3_manager: Optional[S3Manager] = None,
    is_best=False,
):
    """Save comprehensive model checkpoint to local storage or S3"""

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config.to_dict(),
        "dataloader_state": dataloader_state,
        "timestamp": datetime.now().isoformat(),
        "total_batches_processed": epoch
        * len(dataloader_state["loader_state"]["dataset_state"])
        // config.batch_size
        + batch_idx,
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

    config = checkpoint["config"]
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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
        "config": config,
        "dataloader_state": dataloader_state,
    }


def evaluate_model(
    model: LatentActionVQVAE,
    dataloader: PokemonOpenWorldLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    step: int,
    num_batches: int = 10,
    wandb_logger=None,
):
    """Evaluate model on a subset of data"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                video_batch = video_batch.to(device)
                decoded = model(video_batch)
                mse_loss = criterion(video_batch, decoded)
                loss = mse_loss

                total_loss += loss.item() * video_batch.size(0)
                total_samples += video_batch.size(0)

            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    # Log to wandb if available
    if wandb_logger and step is not None:
        wandb_logger.log({"eval/loss": avg_loss}, step=step)

    model.train()  # Switch back to training mode
    return avg_loss


def _reconstruct_predicted_frames(
    video_batch: torch.Tensor, decoded_residuals: torch.Tensor
) -> torch.Tensor:
    """
    Given an input video and predicted residuals between consecutive frames,
    reconstruct the predicted next frames by adding residuals to the original frames.

    video_batch: [B, T, C, H, W]
    decoded_residuals: [B, T-1, C, H, W]
    """
    # Clone to avoid in-place modification of original batch
    predicted = video_batch.clone()
    # For t >= 1, predicted frame t = original frame t-1 + residual_{t-1}
    predicted[:, 1:, :, :, :] = torch.clamp(
        video_batch[:, :-1, :, :, :] + decoded_residuals, 0.0, 1.0
    )
    return predicted


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
    predicted_actions: torch.Tensor,
    decoded_residuals: torch.Tensor,
    config: VideoTrainingConfig,
    prefix: str,
) -> None:
    """
    Save:
      1) Comparison grids of original vs expected next vs predicted next frames
      2) Grids of residual magnitudes to visualize predicted movement.
    """
    save_root = getattr(config, "save_dir", "latent_action_results")

    # Limit number of samples for visualization
    max_samples = 4
    video_batch_vis = video_batch[:max_samples]
    decoded_residuals_vis = decoded_residuals[:max_samples]

    # 1) Next-frame comparisons (original, expected, predicted)
    predicted_full = _reconstruct_predicted_frames(
        video_batch_vis, decoded_residuals_vis
    )
    predicted_videos = convert_video_to_images(predicted_full)
    expected_videos = convert_video_to_images(video_batch_vis)

    comparison_dir = os.path.join(save_root, prefix, "next_frame")
    os.makedirs(comparison_dir, exist_ok=True)
    save_comparison_images_next_frame(
        predicted_videos,
        predicted_actions.squeeze(-1).detach().cpu().numpy().tolist(),
        expected_videos,
        comparison_dir,
    )

    # 2) Residuals-only visualization (movement magnitude)
    # Predicted residuals: use absolute value and scale for visibility, then clamp to [0, 1]
    residual_vis = torch.clamp(decoded_residuals_vis.abs() * 5.0, 0.0, 1.0)
    predicted_residual_videos = convert_video_to_images(residual_vis)

    # Target residuals: ground truth frame-to-frame differences, visualized similarly
    target_residuals = (
        video_batch_vis[:, 1:, :, :, :] - video_batch_vis[:, :-1, :, :, :]
    )
    target_residual_vis = torch.clamp(target_residuals.abs() * 5.0, 0.0, 1.0)
    target_residual_videos = convert_video_to_images(target_residual_vis)

    residual_dir = os.path.join(save_root, prefix, "residuals")
    _save_residual_grid(predicted_residual_videos, target_residual_videos, residual_dir)


def train_epoch(
    model: LatentActionVQVAE,
    train_dataloader: PokemonOpenWorldLoader,
    eval_dataloader: Optional[PokemonOpenWorldLoader],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    config: VideoTrainingConfig,
    s3_manager: Optional[S3Manager] = None,
    start_batch: int = 0,
    wandb_logger=None,
):
    """Train for one epoch with comprehensive logging"""
    total_loss = 0.0
    num_batches = len(train_dataloader)
    best_loss = float("inf")

    # Set up resumable dataloader
    if start_batch > 0:
        train_dataloader.resumable_loader.set_epoch(epoch)
        train_dataloader.resumable_loader.current_batch = start_batch

    epoch_start_time = time.time()
    batch_start_time = time.time()
    commit_beta = 0.2

    for batch_idx, video_batch in enumerate(train_dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass
        decoded, predicted_actions = model(video_batch)

        # Calculate loss (reconstruction loss)
        mse_loss = criterion(video_batch, decoded)
        loss = mse_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # Step scheduler every batch for cosine annealing

        # Periodically save visualizations (comparison grids and residuals)
        if batch_idx % config.save_interval == 0:
            # save_visualizations(
            #     video_batch.detach(),
            #     decoded.detach(),
            #     config,
            #     prefix=f"train/epoch_{epoch}_batch_{batch_idx}",
            # )
            predicted_videos = convert_video_to_images(decoded)
            expected_videos = convert_video_to_images(video_batch)
            save_comparison_images_next_frame(
                predicted_videos,
                predicted_actions.squeeze(-1).detach().cpu().numpy().tolist(),
                expected_videos,
                f"{config.save_dir}/train/epoch_{epoch}_batch_{batch_idx}",
            )

        total_loss += loss.item()
        batch_time = time.time() - batch_start_time

        # Calculate global step
        global_step = epoch * num_batches + batch_idx

        # Log to wandb with system metrics
        if wandb_logger is not None:
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/batch_time": batch_time,
                "train/epoch": epoch,
                "train/batch": batch_idx,
                "train/mse_loss": mse_loss.item(),
            }

            wandb_logger.log(wandb_metrics, step=global_step)

        # Log progress
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                f"Loss: {loss.item():.6f}, LR: {current_lr:.2e}, "
                f"Time: {batch_time:.2f}s"
            )

        # Save checkpoint periodically
        # if batch_idx > 0 and batch_idx % config.save_interval == 0:
        #     dataloader_state = train_dataloader.get_state()
        #     is_best = loss.item() < best_loss
        #     if is_best:
        #         best_loss = loss.item()

        #     save_checkpoint(
        #         model,
        #         optimizer,
        #         scheduler,
        #         epoch,
        #         batch_idx,
        #         loss.item(),
        #         config,
        #         dataloader_state,
        #         config.checkpoint_dir,
        #         s3_manager,
        #         is_best,
        #     )

        # Evaluate periodically
        # if (
        #     eval_dataloader is not None
        #     and batch_idx > 0
        #     and batch_idx % config.eval_interval == 0
        # ):
        #     eval_loss = evaluate_model(
        #         model,
        #         eval_dataloader,
        #         criterion,
        #         device,
        #         wandb_logger=wandb_logger,
        #         step=global_step,
        #     )
        #     logger.info(f"Evaluation loss at batch {batch_idx}: {eval_loss:.6f}")

        # Upload logs to S3 periodically
        if (
            config.use_s3
            and s3_manager
            and batch_idx > 0
            and batch_idx % (config.save_interval * 2) == 0
        ):
            upload_logs_to_s3(config, s3_manager)

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time

    # Log epoch metrics
    # Log epoch summary to wandb
    if wandb_logger is not None:
        epoch_metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_time": epoch_time,
            "train/epoch": epoch,
        }

        wandb_logger.log(epoch_metrics, step=epoch * num_batches)

    logger.info(
        f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s"
    )
    return avg_loss


def setup_wandb(config: VideoTrainingConfig):
    """Initialize Weights & Biases logging"""
    if not config.use_wandb:
        return None

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        group="video-vqvae-test",
        entity=config.wandb_entity,
        name=config.experiment_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=config.to_dict(),
    )

    # Watch the model for gradients and parameters
    return wandb


def main(config: VideoTrainingConfig):
    """Main training function with resumable training support and S3 integration"""

    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"pokemon_vqvae_{timestamp}"

    # Setup wandb
    wandb_logger = setup_wandb(config)

    logger.info(
        f"Starting Pokemon VQVAE training - Experiment: {config.experiment_name}"
    )
    logger.info(f"Using S3: {config.use_s3}")
    logger.info(f"Using Wandb: {config.use_wandb}")

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

    dataset_creator = OpenWorldRunningDatasetCreator(
        dataset_dir=config.frames_dir,
        num_frames_in_video=config.num_images_in_video,
        output_log_json_file_name="log_dir_100000.json",
        local_cache=local_cache,
        limit=100000,
        image_size=config.image_size,
    )

    logger.info("Setting up dataset...")
    train_dataset, test_dataset = dataset_creator.setup(train_percentage=0.9)

    train_dataset = OpenWorldRunningDataset(
        dataset=train_dataset,
        local_cache=local_cache,
        image_size=config.image_size,
    )

    test_dataset = OpenWorldRunningDataset(
        dataset=test_dataset,
        local_cache=local_cache,
        image_size=config.image_size,
    )

    logger.info("Creating data loaders...")
    train_dataloader = PokemonOpenWorldLoader(
        frames_dir=config.frames_dir,
        dataset=train_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
        use_s3=config.use_s3,
        cache_dir=config.local_cache_dir,
        max_cache_size=config.max_cache_size,
    )
    eval_dataloader = PokemonOpenWorldLoader(
        frames_dir=config.frames_dir,
        dataset=test_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
        use_s3=config.use_s3,
        cache_dir=config.local_cache_dir,
        max_cache_size=config.max_cache_size,
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

    # Log dataset info to wandb
    if wandb_logger:
        wandb_logger.log({f"dataset/{key}": value for key, value in train_info.items()})
        wandb_logger.log(
            {f"eval_dataset/{key}": value for key, value in eval_info.items()}
        )

    # Create model
    logger.info(f"Creating model on device {device}...")
    model = create_model(config)
    model.to(device)

    # Watch model with wandb
    if wandb_logger:
        wandb_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    logger.info(f"Optimizer created with learning rate: {config.learning_rate}")

    # Cosine annealing scheduler
    total_steps = config.num_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.min_learning_rate
    )

    # Loss function
    criterion = next_frame_reconstruction_loss
    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0

    if config.resume_from:
        checkpoint_info = load_checkpoint(
            config.resume_from, optimizer, scheduler, device, s3_manager
        )
        if checkpoint_info:
            start_epoch = checkpoint_info["epoch"]
            start_batch = checkpoint_info.get("batch_idx", 0)

            # Restore dataloader state
            if "dataloader_state" in checkpoint_info:
                dataloader_state = checkpoint_info["dataloader_state"]
                train_dataloader.resumable_loader = (
                    train_dataloader.create_resumable_loader(start_epoch, start_batch)
                )

    # Training loop
    logger.info("Starting training loop...")
    best_loss = float("inf")

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss = train_epoch(
                model,
                train_dataloader,
                eval_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                config,
                s3_manager,
                epoch_start_batch,
                wandb_logger,
            )

            # # Save end-of-epoch checkpoint
            # is_best = avg_loss < best_loss
            # if is_best:
            #     best_loss = avg_loss

            #     save_checkpoint(
            #         model,
            #         optimizer,
            #         scheduler,
            #         epoch,
            #         len(train_dataloader),
            #         avg_loss,
            #         config,
            #         train_dataloader.get_state(),
            #         config.checkpoint_dir,
            #         s3_manager,
            #         False,
            #     )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interruption
        dataloader_state = train_dataloader.get_state()
        # save_checkpoint(
        #     model,
        #     optimizer,
        #     scheduler,
        #     epoch,
        #     train_dataloader.resumable_loader.current_batch,
        #     avg_loss,
        #     config,
        #     dataloader_state,
        #     config.checkpoint_dir,
        #     s3_manager,
        # )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        # Finish wandb run
        if wandb_logger:
            wandb_logger.finish()

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

    if config.use_wandb:
        logger.info(f"Training metrics logged to Wandb project: {config.wandb_project}")


if __name__ == "__main__":
    config = VideoTrainingConfig.from_cli()

    logger.info(f"Starting training... config: {config.to_dict()}")
    main(config)
    logger.info("Training completed!")
