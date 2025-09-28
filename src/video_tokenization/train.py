# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
import time
from datetime import datetime
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from data_collection.pokemon_frame_loader import PokemonFrameLoader
from latent_action_model.training_args import VideoTrainingConfig
from loss.loss_fns import reconstruction_residual_loss
from s3.s3_utils import S3Manager, default_s3_manager
from video_tokenization.tokenizer import VideoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def create_model(config: VideoTrainingConfig):
    """Create and initialize the VQVAE model"""
    model = VideoTokenizer(
        channels=3,
        image_height=config.image_size,
        image_width=config.image_size,
        patch_height=config.patch_size,
        patch_width=config.patch_size,
        num_images_in_video=config.num_images_in_video,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        embedding_dim=config.latent_dim,
    )

    return model

def save_checkpoint(model: VideoTokenizer, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.CosineAnnealingLR, epoch: int,
        batch_idx: int, loss: float, config: VideoTrainingConfig, best_loss: float, dataloader_state: dict):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'best_loss': best_loss,
        'config': config.to_dict(),
        'dataloader_state': dataloader_state,
    }
    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}_batch{batch_idx}.pt')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

def load_checkpoint(checkpoint_path: str, model: VideoTokenizer, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.CosineAnnealingLR, device: torch.device) -> tuple[VideoTokenizer, optim.Optimizer, optim.lr_scheduler.CosineAnnealingLR, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    dataloader_state = PokemonFrameLoader.load_state(checkpoint['dataloader_state'])
    return model, optimizer, scheduler, dataloader_state


def train_epoch(
        model: VideoTokenizer, dataloader: PokemonFrameLoader, optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.CosineAnnealingLR,
        criterion: Callable[[torch.Tensor, torch.Tensor],
                            torch.Tensor],
        device: torch.device, epoch: int, config: VideoTrainingConfig,
        s3_manager: Optional[S3Manager] = None, start_batch: int = 0, wandb_logger=None):
    total_loss = 0.0
    num_batches = len(dataloader)
    best_loss = float('inf')

    # Set up resumable dataloader
    if start_batch > 0:
        dataloader.resumable_loader.set_epoch(epoch)
        dataloader.resumable_loader.current_batch = start_batch

    epoch_start_time = time.time()
    batch_start_time = time.time()

    for batch_idx, video_batch in enumerate(dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass with decoder
        # Outputs a tensor of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        decoded = model(video_batch)

        # Calculate loss (reconstruction loss)
        # TODO: Consider patch level loss
        loss = criterion(video_batch, decoded)

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
                'train/loss': loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/batch_time': batch_time,
                'train/epoch': epoch,
                'train/batch': batch_idx,
            }

            wandb_logger.log(wandb_metrics, step=global_step)

        # Log progress
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                f'Loss: {loss.item():.6f}, LR: {current_lr:.2e}, '
                f'Time: {batch_time:.2f}s'
            )

        # Save checkpoint periodically
        if batch_idx > 0 and batch_idx % config.save_interval == 0:
            dataloader_state = dataloader.get_state()
            is_best = loss < best_loss
            if is_best:
                best_loss = loss.item()

            save_checkpoint(
                model, optimizer, scheduler, epoch, batch_idx,
                loss.item(), config, best_loss, dataloader_state
            )

        # # Evaluate periodically
        # if batch_idx > 0 and batch_idx % config.eval_interval == 0:
        #     eval_loss = evaluate_model(
        #         model, dataloader, criterion, device,
        #         wandb_logger=wandb_logger, step=global_step
        #     )
        #     logger.info(f'Evaluation loss at batch {batch_idx}: {eval_loss:.6f}')

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time

    # Log epoch metrics
    # Log epoch summary to wandb
    if wandb_logger:
        epoch_metrics = {
            'train/epoch_loss': avg_loss,
            'train/epoch_time': epoch_time,
            'train/epoch': epoch,
        }

        wandb_logger.log(epoch_metrics, step=epoch * num_batches)

    logger.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s')
    return avg_loss


def setup_wandb(config: VideoTrainingConfig):
    """Initialize Weights & Biases logging"""
    if not config.use_wandb:
        return None

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        group="video-tokenizer-test",
        entity=config.wandb_entity,
        name=config.experiment_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=config.to_dict()
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

    logger.info(f"Starting Pokemon VQVAE training - Experiment: {config.experiment_name}")
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
        limit=100
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Log dataset info to wandb
    if wandb_logger:
        wandb_logger.log({f'dataset/{key}': value for key, value in info.items()})

    # Create model
    logger.info(f"Creating model on device {device}...")
    model = create_model(config)
    model.to(device)

    logger.info(f"Num params: {sum(p.numel() for p in model.parameters())} on device {device}")

    # Watch model with wandb
    if wandb_logger:
        wandb_logger.watch(model, log='all', log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    logger.info(f"Optimizer created with learning rate: {config.learning_rate}")

    # Cosine annealing scheduler
    total_steps = config.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.min_learning_rate
    )

    # Loss function
    criterion = reconstruction_residual_loss
    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0

    if config.resume_from:
        checkpoint_info = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, device
        )
        if checkpoint_info:
            start_epoch = checkpoint_info['epoch']
            start_batch = checkpoint_info.get('batch_idx', 0)

            # Restore dataloader state
            if 'dataloader_state' in checkpoint_info:
                dataloader_state = checkpoint_info['dataloader_state']
                dataloader.resumable_loader = dataloader.create_resumable_loader(
                    start_epoch, start_batch
                )

    # Training loop
    logger.info("Starting training loop...")
    best_loss = float('inf')

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss = train_epoch(
                model, dataloader, optimizer, scheduler, criterion, device,
                epoch, config, s3_manager, epoch_start_batch, wandb_logger
            )

            save_checkpoint(
                model, optimizer, scheduler, epoch, len(dataloader),
                avg_loss, config, dataloader.get_state(),
                config.checkpoint_dir, s3_manager, False
            )


    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interruption
        dataloader_state = dataloader.get_state()
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            dataloader.resumable_loader.current_batch,
            avg_loss, config, dataloader_state, config.checkpoint_dir, s3_manager
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
        if config._temp_tensorboard_dir and os.path.exists(config._temp_tensorboard_dir):
            import shutil
            shutil.rmtree(config._temp_tensorboard_dir)

    logger.info("Training completed!")
    logger.info(f"Best loss achieved: {best_loss:.6f}")

    if config.use_s3 and s3_manager:
        logger.info(f"Checkpoints and logs saved to S3 bucket: {s3_manager.bucket_name}")
    else:
        logger.info(
            f"Tensorboard logs saved to: {os.path.join(config.tensorboard_dir, config.experiment_name or 'default')}")

    if config.use_wandb:
        logger.info(f"Training metrics logged to Wandb project: {config.wandb_project}")


if __name__ == "__main__":
    config = VideoTrainingConfig.from_cli()

    logger.info(f"Starting training... config: {config.to_dict()}")
    main(config)
    logger.info("Training completed!")
