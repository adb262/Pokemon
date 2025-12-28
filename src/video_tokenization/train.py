# python -m src.video_tokenization.train --use_s3 true --frames_dir pokemon --seed_cache --num_images_in_video 2 --batch_size 2
# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
import time
from datetime import datetime
from typing import Callable

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.data_loaders.pokemon_open_world_loader import PokemonOpenWorldLoader
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)
from data.s3.s3_utils import default_s3_manager
from loss.loss_fns import reconstruction_loss
from monitoring.setup_wandb import setup_wandb
from video_tokenization.checkpoints import save_checkpoint
from video_tokenization.create_tokenizer import create_model
from video_tokenization.eval import (
    convert_video_to_images,
    eval_model,
    save_comparison_images,
)
from video_tokenization.tokenizer import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig
from wandb.wandb_run import Run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def train_epoch(
    model: VideoTokenizer,
    dataloader: PokemonOpenWorldLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.CosineAnnealingLR,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    config: VideoTokenizerTrainingConfig,
    wandb_logger: Run,
    start_batch: int = 0,
    save_dir: str = "tokenization_results",
):
    total_loss = 0.0
    num_batches = len(dataloader)
    best_loss = float("inf")

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
        loss = criterion(video_batch, decoded)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if batch_idx % config.save_interval == 0:
            images = convert_video_to_images(video_batch)
            decoded_images = convert_video_to_images(decoded)
            save_comparison_images(
                decoded_images,
                images,
                f"{save_dir}/train/epoch_{epoch}_batch_{batch_idx}",
            )

        optimizer.step()
        scheduler.step()  # Step scheduler every batch for cosine annealing

        total_loss += loss.item()
        batch_time = time.time() - batch_start_time

        # Calculate global step
        global_step = epoch * num_batches + batch_idx

        # Log to wandb with system metrics
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
            is_best = loss < best_loss
            if is_best:
                best_loss = loss.item()

            predicted_videos = convert_video_to_images(decoded)
            expected_videos = convert_video_to_images(video_batch)
            save_comparison_images(
                predicted_videos,
                expected_videos,
                f"{save_dir}/train/epoch_{epoch}_batch_{batch_idx}",
            )

            # save_checkpoint(
            #     model,
            #     optimizer,
            #     scheduler,
            #     epoch,
            #     batch_idx,
            #     loss.item(),
            #     config,
            #     best_loss,
            #     dataloader_state,
            # )

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


def main(config: VideoTokenizerTrainingConfig):
    """Main training function with resumable training support and S3 integration"""

    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"pokemon_vqvae_{timestamp}"

    # Setup wandb
    wandb_logger = setup_wandb(
        project=config.wandb_project,
        group="video-tokenizer-test",
        entity=config.wandb_entity,
        name=config.experiment_name,
        tags=config.wandb_tags or [],
        notes=config.wandb_notes or "",
        config=config.to_dict(),
    )

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
        dataset_dir="saved_datasets",
        num_frames_in_video=config.num_images_in_video,
        output_log_json_file_name="log_dir_100000.json",
        local_cache=local_cache,
        limit=100000,
        image_size=config.image_size,
    )

    logger.info("Setting up dataset...")
    train_dataset, test_dataset = dataset_creator.setup(train_percentage=0.9)

    # Lets overfit
    # test_dataset = train_dataset

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

    logger.info(f"Creating data loader with {len(train_dataset)} videos...")
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
    test_dataloader = PokemonOpenWorldLoader(
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
    test_info = test_dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in train_info.items():
        logger.info(f"  {key}: {value}")

    for key, value in test_info.items():
        logger.info(f"  {key}: {value}")

    # Log dataset info to wandb
    if wandb_logger:
        wandb_logger.log({f"dataset/{key}": value for key, value in train_info.items()})
        wandb_logger.log(
            {f"test_dataset/{key}": value for key, value in test_info.items()}
        )

    # Create model
    logger.info(f"Creating model on device {device}...")
    model = create_model(config)
    model.to(device)

    logger.info(
        f"Num params: {sum(p.numel() for p in model.parameters())} on device {device}"
    )

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
    criterion = reconstruction_loss
    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0
    # Training loop
    logger.info("Starting training loop...")
    best_loss = float("inf")

    # first, evaluate on test dataset
    test_loss = eval_model(
        model,
        test_dataloader,
        criterion,
        device,
        epoch=0,
        wandb_logger=wandb_logger,
        save_dir=config.save_dir,
    )
    logger.info(f"Test loss: {test_loss:.6f}")

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                config,
                wandb_logger,
                epoch_start_batch,
                config.save_dir,
            )

            # eval_loss = eval_model(
            #     model,
            #     test_dataloader,
            #     criterion,
            #     device,
            #     epoch,
            #     wandb_logger=wandb_logger,
            #     save_dir=config.save_dir,
            # )
            # logger.info(f"Test loss: {eval_loss:.6f}")

            # if eval_loss < best_loss:
            #     best_loss = eval_loss

            #     save_checkpoint(
            #         model,
            #         optimizer,
            #         scheduler,
            #         epoch,
            #         len(train_dataloader),
            #         avg_loss,
            #         config,
            #         best_loss,
            #         train_dataloader.get_state(),
            #     )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interruption
        dataloader_state = train_dataloader.get_state()
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_dataloader.resumable_loader.current_batch,
            avg_loss,
            config,
            best_loss,
            dataloader_state,
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
    config = VideoTokenizerTrainingConfig.from_cli()

    logger.info(f"Starting training... config: {config.to_dict()}")
    main(config)
    logger.info("Training completed!")
