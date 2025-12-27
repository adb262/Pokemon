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
from data_collection.pokemon_frame_loader import PokemonFrameLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from data.s3.s3_utils import S3Manager, default_s3_manager
from latent_action_model.laqa import LatentActionQuantization
from latent_action_model.training_args import VideoTrainingConfig
from loss.loss_fns import next_frame_reconstruction_residual_loss

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
    model = LatentActionQuantization(
        dim=config.d_model,
        quant_dim=config.latent_dim,
        codebook_size=config.num_embeddings,
        image_size=config.image_size,
        patch_size=config.patch_size,
        spatial_depth=config.num_transformer_layers,
        temporal_depth=config.num_transformer_layers,
        channels=3,
    )

    return model


def save_checkpoint(
    model: LatentActionQuantization,
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
    model: LatentActionQuantization,
    dataloader: PokemonFrameLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    num_batches: int = 10,
    wandb_logger=None,
    step=None,
):
    """Evaluate model on a subset of data"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (image1_batch, image2_batch) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            image1_batch = image1_batch.to(device)
            image2_batch = image2_batch.to(device)

            try:
                decoded = model(image1_batch, image2_batch)
                loss = criterion(image1_batch, image2_batch, decoded)

                total_loss += loss.item() * image1_batch.size(0)
                total_samples += image1_batch.size(0)

            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    # Log to wandb if available
    if wandb_logger and step is not None:
        wandb_logger.log({"eval/loss": avg_loss}, step=step)

    model.train()  # Switch back to training mode
    return avg_loss


def train_epoch(
    model: LatentActionQuantization,
    dataloader: PokemonFrameLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    config: VideoTrainingConfig,
    s3_manager: Optional[S3Manager] = None,
    start_batch: int = 0,
    wandb_logger=None,
):
    """Train for one epoch with comprehensive logging"""
    total_loss = 0.0
    num_batches = len(dataloader)
    best_loss = float("inf")

    # Set up resumable dataloader
    if start_batch > 0:
        dataloader.resumable_loader.set_epoch(epoch)
        dataloader.resumable_loader.current_batch = start_batch

    epoch_start_time = time.time()
    batch_start_time = time.time()
    commit_beta = 0.2

    for batch_idx, video_batch in enumerate(dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass
        loss, num_unique_indices = model(video_batch.permute(0, 2, 1, 3, 4))

        # Calculate loss (reconstruction loss)
        # mse_loss = criterion(video_batch, recon_loss) /
        # loss = mse_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # Step scheduler every batch for cosine annealing

        total_loss += loss.item()
        batch_time = time.time() - batch_start_time

        # Calculate global step
        global_step = epoch * num_batches + batch_idx

        # Log to wandb with system metrics
        if wandb_logger:
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/batch_time": batch_time,
                "train/epoch": epoch,
                "train/batch": batch_idx,
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
        if batch_idx > 0 and batch_idx % config.save_interval == 0:
            dataloader_state = dataloader.get_state()
            is_best = loss.item() < best_loss
            if is_best:
                best_loss = loss.item()

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                batch_idx,
                loss.item(),
                config,
                dataloader_state,
                config.checkpoint_dir,
                s3_manager,
                is_best,
            )

        batch_end_time = time.time()

        # Evaluate periodically
        # if batch_idx > 0 and batch_idx % config.eval_interval == 0:
        #     eval_loss = evaluate_model(
        #         model, dataloader, criterion, device,
        #         wandb_logger=wandb_logger, step=global_step
        #     )
        #     logger.info(f'Evaluation loss at batch {batch_idx}: {eval_loss:.6f}')

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
    if wandb_logger:
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
    dataloader = PokemonFrameLoader(
        frames_dir=config.frames_dir,
        batch_size=config.batch_size,
        num_frames_in_video=config.num_images_in_video,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
        use_s3=config.use_s3,
        cache_dir=config.local_cache_dir,
        max_cache_size=config.max_cache_size,
        stage="train",
        seed_cache=config.seed_cache,
        limit=10000,
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Log dataset info to wandb
    if wandb_logger:
        wandb_logger.log({f"dataset/{key}": value for key, value in info.items()})

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
    total_steps = config.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.min_learning_rate
    )

    # Loss function
    criterion = next_frame_reconstruction_residual_loss
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
                dataloader.resumable_loader = dataloader.create_resumable_loader(
                    start_epoch, start_batch
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
                dataloader,
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

            # Save end-of-epoch checkpoint
            # dataloader_state = dataloader.get_state()
            # is_best = avg_loss < best_loss
            # if is_best:
            #     best_loss = avg_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                len(dataloader),
                avg_loss,
                config,
                dataloader.get_state(),
                config.checkpoint_dir,
                s3_manager,
                False,
            )

            # # Reset start_batch for subsequent epochs
            # start_batch = 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interruption
        dataloader_state = dataloader.get_state()
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            dataloader.resumable_loader.current_batch,
            avg_loss,
            config,
            dataloader_state,
            config.checkpoint_dir,
            s3_manager,
        )
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
