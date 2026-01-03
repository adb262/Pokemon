import logging
import os
import time
from datetime import datetime
from typing import Callable

import torch
import tyro

from data.data_loaders.pokemon_open_world_loader import PokemonOpenWorldLoader
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)
from loss.loss_fns import next_frame_reconstruction_loss
from monitoring.frechet_distance import compute_frechet_distance
from monitoring.setup_wandb import setup_wandb
from monitoring.videos import convert_video_to_images, save_comparison_images
from monitoring.wandb_media import log_image_batches
from video_tokenization.checkpoints import load_checkpoint
from video_tokenization.create_tokenizer import create_model
from video_tokenization.tokenizer import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig
from wandb.wandb_run import Run

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval_model(
    model: VideoTokenizer,
    dataloader: PokemonOpenWorldLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int | str,
    wandb_logger: Run,
    save_dir: str = "tokenization_results",
    global_step: int | None = None,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    saved_image_paths: list[str] = []

    # Collect real and predicted next frames across the eval set
    real_next_frames_batches: list[torch.Tensor] = []
    pred_next_frames_batches: list[torch.Tensor] = []

    eval_dir = f"{save_dir}/eval/epoch_{epoch}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving eval results to {eval_dir}")

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            video_batch = video_batch.to(device)

            decoded: torch.Tensor = model(video_batch)
            loss = criterion(video_batch, decoded)

            # Predicting full frames. Video VQVAE is not looking ahead.
            if decoded.dim() == 5 and video_batch.dim() == 5:
                real_next_frames_batches.append(video_batch.detach().cpu())
                pred_next_frames_batches.append(decoded.detach().cpu())

            # Save the image locally
            # View as (b, num_frames, h, w, c)
            # Decoded is the residual, must add the previous frame to get the predicted frame
            predicted_videos = convert_video_to_images(decoded)
            expected_videos = convert_video_to_images(video_batch)
            image_path = f"{eval_dir}/batch_{batch_idx}/comparison_grid.png"
            save_comparison_images(predicted_videos, expected_videos, image_path)
            saved_image_paths.append(image_path)
            logger.info(f"Saved comparison image to {image_path}")

            total_loss += loss.item() * video_batch.size(0)
            total_samples += video_batch.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    logger.info(
        f"Eval complete: avg_loss={avg_loss:.6f}, saved {len(saved_image_paths)} images"
    )

    if real_next_frames_batches and pred_next_frames_batches:
        real_all = torch.cat(real_next_frames_batches, dim=0)
        pred_all = torch.cat(pred_next_frames_batches, dim=0)
        logger.info(
            f"Computing Frechet distance between {real_all.shape} and {pred_all.shape}"
        )
        t = time.time()
        frechet_distance = compute_frechet_distance(real_all, pred_all)
        logger.info(f"Frechet distance computed in {time.time() - t} seconds")
    else:
        frechet_distance = float("inf")

    if wandb_logger:
        # Use consistent metric names for plotting over time
        log_dict: dict = {
            "eval/loss": avg_loss,
            "eval/frechet_distance": frechet_distance,
            "eval/epoch": epoch if isinstance(epoch, int) else 0,
        }

        # Log with global_step if provided for proper x-axis alignment with training
        if global_step is not None:
            wandb_logger.log(log_dict, step=global_step)
        else:
            wandb_logger.log(log_dict)

        # Log comparison images in batches of 5, stacked vertically
        log_image_batches(
            wandb_logger,
            key_prefix="eval/comparison",
            image_paths=saved_image_paths,
            batch_size=5,
            step=global_step,
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
        entity=config.wandb_entity or "",
        name=config.experiment_name,
        tags=config.wandb_tags or [],
        notes=config.wandb_notes or "",
        config=config.__dict__,
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
        dataset_dir=config.frames_dir,
        num_frames_in_video=config.num_images_in_video,
        output_log_json_file_name="log_dir_10000.json",
        local_cache=local_cache,
        limit=10000,
        image_size=config.image_size,
        use_s3=config.use_s3,
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

    logger.info("Creating data loader...")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )
    model, optimizer, scheduler = load_checkpoint(
        "checkpoints/checkpoint_epoch4_batch405.pt", model, optimizer, scheduler, device
    )
    model.to(device)
    criterion = next_frame_reconstruction_loss

    # first, evaluate on test dataset
    test_loss = eval_model(
        model,
        test_dataloader,
        criterion,
        device,
        epoch=0,
        wandb_logger=wandb_logger,
        save_dir="eval_results",
    )
    logger.info(f"Test loss: {test_loss:.6f}")

    try:
        eval_loss = eval_model(
            model,
            test_dataloader,
            criterion,
            device,
            epoch="TEST_EVAL",
            wandb_logger=wandb_logger,
            save_dir="eval_results",
        )
        logger.info(f"Test loss: {eval_loss:.6f}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        raise
    finally:
        logger.info("Evaluation completed!")
    logger.info(f"Evaluation loss: {eval_loss:.6f}")


if __name__ == "__main__":
    config = tyro.cli(VideoTokenizerTrainingConfig)

    logger.info(f"Starting training... config: {config.__dict__}")
    main(config)
    logger.info("Training completed!")
