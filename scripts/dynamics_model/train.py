# python -m scripts.mask_git.train --tokenizer_checkpoint_path checkpoints/tokenizer.pt --frames_dir pokemon --num_images_in_video 4 --batch_size 4
import logging
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.data_loaders.pokemon_open_world_loader import PokemonOpenWorldLoader
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)
from data.s3.s3_utils import default_s3_manager
from dynamics_model.checkpoints import save_checkpoint
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.model import DynamicsModel
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from latent_action_model.model import LatentActionVQVAE
from monitoring.frechet_distance import compute_frechet_distance, compute_fvd
from monitoring.setup_wandb import setup_wandb
from monitoring.videos import convert_video_to_images, save_comparison_images_next_frame
from monitoring.wandb_media import log_image_batches
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: DynamicsModel,
    dataloader: PokemonOpenWorldLoader,
    device: torch.device,
    epoch: int,
    global_step: int,
    wandb_logger=None,
    save_dir: str = "dynamics_model_results",
    num_batches: int = 10,
) -> tuple[float, float]:
    """Evaluate model on a subset of data with FID/FVD metrics and comparison images."""
    model.eval()
    total_token_loss = 0.0
    total_action_loss = 0.0
    total_samples = 0
    saved_image_paths: list[str] = []

    # Collect real and reconstructed frames for FID/FVD computation
    real_frames_batches: list[torch.Tensor] = []
    pred_frames_batches: list[torch.Tensor] = []

    eval_dir = f"{save_dir}/eval/epoch_{epoch}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving eval results to {eval_dir}")

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                video_batch = video_batch.to(device)
                _, token_loss, action_loss = model(video_batch)

                total_token_loss += token_loss.item() * video_batch.size(0)
                total_action_loss += action_loss.item() * video_batch.size(0)
                total_samples += video_batch.size(0)

                # Get reconstructed video from action model for FID/FVD
                # The action model reconstructs frames 1:T from frames 0:T-1
                action_encoded = model.action_model.encode(video_batch)
                action_tokens = model.action_model.get_action_sequence(action_encoded)
                reconstructed_video = model.action_model.decode(action_tokens)

                # Real target frames: video[:, 1:, :, :, :]
                real_target_frames = video_batch[:, 1:, :, :, :]
                real_frames_batches.append(real_target_frames.detach().cpu())
                pred_frames_batches.append(reconstructed_video.detach().cpu())

                # Save comparison images
                predicted_videos = convert_video_to_images(reconstructed_video)
                expected_videos = convert_video_to_images(real_target_frames)
                image_path = f"{eval_dir}/batch_{batch_idx}/comparison_grid.png"
                save_comparison_images_next_frame(
                    predicted_videos,
                    action_tokens.squeeze(-1).detach().cpu().numpy().tolist(),
                    expected_videos,
                    image_path,
                )
                saved_image_paths.append(image_path)
                logger.debug(f"Saved comparison image to {image_path}")

            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_token_loss = (
        total_token_loss / total_samples if total_samples > 0 else float("inf")
    )
    avg_action_loss = (
        total_action_loss / total_samples if total_samples > 0 else float("inf")
    )

    # Compute FID and FVD
    frechet_distance = float("inf")
    fvd_score = float("inf")

    if real_frames_batches and pred_frames_batches:
        real_all = torch.cat(real_frames_batches, dim=0)
        pred_all = torch.cat(pred_frames_batches, dim=0)

        # Compute FID (Frechet Inception Distance) - frame-level metric
        logger.info(f"Computing FID between {real_all.shape} and {pred_all.shape}")
        t = time.time()
        try:
            frechet_distance = compute_frechet_distance(real_all, pred_all)
            logger.info(
                f"FID computed in {time.time() - t:.2f} seconds: {frechet_distance:.4f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute FID: {e}")

        # Compute FVD (Frechet Video Distance) - video-level metric
        logger.info(f"Computing FVD between {real_all.shape} and {pred_all.shape}")
        t = time.time()
        try:
            fvd_score = compute_fvd(real_all, pred_all)
            logger.info(
                f"FVD computed in {time.time() - t:.2f} seconds: {fvd_score:.4f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute FVD: {e}")

    logger.info(
        f"Eval complete: token_loss={avg_token_loss:.6f}, action_loss={avg_action_loss:.6f}, "
        f"FID={frechet_distance:.4f}, FVD={fvd_score:.4f}, saved {len(saved_image_paths)} images"
    )

    # Log to wandb if available
    if wandb_logger and global_step is not None:
        log_dict = {
            "eval/token_loss": avg_token_loss,
            "eval/action_loss": avg_action_loss,
            "eval/total_loss": avg_token_loss + avg_action_loss,
            "eval/fid": frechet_distance,
            "eval/fvd": fvd_score,
            "eval/epoch": epoch,
        }
        wandb_logger.log(log_dict, step=global_step)

        # Log comparison images in batches of 5, stacked vertically
        if saved_image_paths:
            log_image_batches(
                wandb_logger,
                key_prefix="eval/comparison",
                image_paths=saved_image_paths,
                batch_size=5,
                step=global_step,
            )

    model.train()
    return avg_token_loss, avg_action_loss


def train_epoch(
    dynamics_model: DynamicsModel,
    action_model: LatentActionVQVAE,
    tokenizer: VideoTokenizer,
    train_dataloader: PokemonOpenWorldLoader,
    test_dataloader: PokemonOpenWorldLoader,
    dynamics_optimizer: optim.Optimizer,
    dynamics_scheduler: optim.lr_scheduler.LRScheduler,
    action_optimizer: optim.Optimizer,
    action_scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: DynamicsModelTrainingConfig,
    wandb_logger,
    start_batch: int = 0,
    save_dir: str = "mask_git_results",
):
    """Train for one epoch"""
    total_loss = 0.0
    num_batches = len(train_dataloader)
    best_loss = float("inf")
    accumulation_steps = config.gradient_accumulation_steps

    # Set up resumable dataloader
    if start_batch > 0:
        train_dataloader.resumable_loader.set_epoch(epoch)
        train_dataloader.resumable_loader.current_batch = start_batch

    os.makedirs(f"{save_dir}/train/epoch_{epoch}", exist_ok=True)

    epoch_start_time = time.time()
    action_loss, token_loss = [], []

    for batch_idx, video_batch in enumerate(train_dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass - MaskGIT returns (predictions, token_loss, action_loss)
        decoded, token_loss, action_loss = dynamics_model(video_batch)

        # Backward pass (accumulate gradients)
        token_loss.backward()
        action_loss.backward()
        token_loss.append(token_loss.item())
        action_loss.append(action_loss.item())

        # Only step optimizer after accumulating enough gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            # Gradient clipping
            if config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    dynamics_model.parameters(), max_norm=config.gradient_clipping
                )
                torch.nn.utils.clip_grad_norm_(
                    action_model.parameters(), max_norm=config.gradient_clipping
                )

            dynamics_optimizer.step()
            action_optimizer.step()

            dynamics_scheduler.step()
            action_scheduler.step()

            dynamics_optimizer.zero_grad()
            action_optimizer.zero_grad()

            avg_token_loss = sum(token_loss) / len(token_loss)
            avg_action_loss = sum(action_loss) / len(action_loss)
            total_loss = avg_token_loss + avg_action_loss

            batch_time = time.time() - batch_start_time

            # Calculate global step
            global_step = epoch * (num_batches // accumulation_steps) + (
                batch_idx // accumulation_steps
            )

            # Log to wandb
            if wandb_logger:
                wandb_metrics = {
                    "train/loss": total_loss,
                    "train/token_loss": avg_token_loss,
                    "train/action_loss": avg_action_loss,
                    "train/learning_rate": dynamics_scheduler.get_last_lr()[0],
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                }
                wandb_logger.log(wandb_metrics, step=global_step)

            # Log progress
            if (batch_idx // accumulation_steps) % config.log_interval == 0:
                current_lr = dynamics_scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {total_loss:.6f} (token: {avg_token_loss:.6f}, action: {avg_action_loss:.6f}), "
                    f"LR: {current_lr:.2e}, Time: {batch_time:.2f}s"
                )

            action_loss, token_loss = [], []

            # Save checkpoint periodically
            optimizer_step = batch_idx // accumulation_steps
            if optimizer_step > 0 and optimizer_step % config.save_interval == 0:
                dataloader_state = train_dataloader.get_state()
                is_best = total_loss < best_loss
                if is_best:
                    best_loss = total_loss

                # Evaluate on test set
                eval_token_loss, eval_action_loss = evaluate_model(
                    dynamics_model,
                    test_dataloader,
                    device,
                    epoch,
                    global_step,
                    wandb_logger=wandb_logger,
                    save_dir=config.save_dir,
                )
                logger.info(
                    f"Eval - Token Loss: {eval_token_loss:.6f}, Action Loss: {eval_action_loss:.6f}"
                )

                save_checkpoint(
                    dynamics_model,
                    dynamics_optimizer,
                    dynamics_scheduler,
                    epoch,
                    batch_idx,
                    total_loss,
                    config,
                    best_loss,
                    dataloader_state,
                )

    # Calculate average loss over optimizer steps
    num_optimizer_steps = num_batches // accumulation_steps
    avg_loss = total_loss / max(num_optimizer_steps, 1)
    epoch_time = time.time() - epoch_start_time

    # Log epoch summary
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
    return avg_loss, global_step


def main(config: DynamicsModelTrainingConfig):
    """Main training function"""

    # Validate required config
    if not config.tokenizer_checkpoint_path:
        raise ValueError(
            "tokenizer_checkpoint_path is required to load pretrained tokenizer"
        )

    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"maskgit_{timestamp}"

    # Setup wandb
    wandb_logger = setup_wandb(
        project=config.wandb_project,
        group="mask-git-training",
        entity=config.wandb_entity,
        name=config.experiment_name,
        tags=config.wandb_tags or [],
        notes=config.wandb_notes or "",
        config=config.__dict__,
    )

    logger.info(f"Starting MaskGIT training - Experiment: {config.experiment_name}")
    logger.info(f"Using S3: {config.use_s3}")
    logger.info(f"Using Wandb: {config.use_wandb}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Create device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Load pretrained tokenizer
    logger.info(
        f"Loading pretrained tokenizer from: {config.tokenizer_checkpoint_path}"
    )
    tokenizer, tokenizer_config = load_model_from_checkpoint(
        config.tokenizer_checkpoint_path, device
    )
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    logger.info(f"Tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")

    # Create action model (will be co-trained)
    logger.info("Creating action model for co-training...")
    action_model = create_action_model_from_dynamics_config(config)
    action_model.to(device)
    logger.info(
        f"Action model params: {sum(p.numel() for p in action_model.parameters())}"
    )

    # Create MaskGIT model
    logger.info("Creating MaskGIT model...")
    model = create_dynamics_model(config, tokenizer, action_model)
    model.to(device)
    logger.info(f"MaskGIT total params: {sum(p.numel() for p in model.parameters())}")
    logger.info(
        f"MaskGIT trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

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
        output_log_json_file_name="log_dir_50000.json",
        local_cache=local_cache,
        limit=50000,
        image_size=config.image_size,
        use_s3=config.use_s3,
    )

    if config.dataset_train_key is None:
        logger.info("Setting up dataset...")
        train_dataset, test_dataset = dataset_creator.setup(train_percentage=0.9)
    else:
        logger.info(f"Loading dataset from {config.dataset_train_key}")
        train_dataset = dataset_creator.load_existing_dataset(config.dataset_train_key)
        test_dataset = dataset_creator.load_existing_dataset(
            config.dataset_train_key.replace("train", "test")
        )

        if config.sync_from_s3:
            logger.info("Syncing dataset from S3...")
            dataset_creator.ensure_files_exist(train_dataset)
            dataset_creator.ensure_files_exist(test_dataset)

    train_dataset = OpenWorldRunningDataset(
        dataset=train_dataset,
        local_cache=local_cache,
        image_size=config.image_size,
    )

    test_dataset = OpenWorldRunningDataset(
        dataset=test_dataset,
        local_cache=local_cache,
        image_size=config.image_size,
        limit=100,
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
    logger.info("Dataset Info (train):")
    for key, value in train_info.items():
        logger.info(f"  {key}: {value}")
    logger.info("Dataset Info (test):")
    for key, value in test_info.items():
        logger.info(f"  {key}: {value}")

    # Log dataset info to wandb
    if wandb_logger:
        wandb_logger.log(
            {f"dataset/{key}": value for key, value in train_info.items()},
            commit=False,
        )
        wandb_logger.log(
            {f"test_dataset/{key}": value for key, value in test_info.items()},
            commit=False,
        )
        wandb_logger.log(
            {
                "config/effective_batch_size": config.batch_size
                * config.gradient_accumulation_steps,
                "config/gradient_accumulation_steps": config.gradient_accumulation_steps,
            },
            commit=False,
        )

    # Watch model with wandb
    if wandb_logger:
        wandb_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.dynamics_learning_rate, weight_decay=1e-4
    )
    action_optimizer = optim.AdamW(
        action_model.parameters(), lr=config.action_learning_rate, weight_decay=1e-4
    )
    logger.info(
        f"Optimizer created with learning rate: {config.dynamics_learning_rate}"
    )

    # Cosine annealing scheduler with warmup
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_steps

    # Linear warmup from 0 to learning_rate over warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    action_warmup_scheduler = LinearLR(
        action_optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Cosine annealing for the remaining steps
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.min_learning_rate,
    )
    action_cosine_scheduler = CosineAnnealingLR(
        action_optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.min_learning_rate,
    )

    action_scheduler = SequentialLR(
        action_optimizer,
        schedulers=[action_warmup_scheduler, action_cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Combine warmup + cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total optimizer steps: {total_steps}")

    # Log effective batch size
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Effective batch size: {effective_batch_size}")

    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0
    best_loss = float("inf")

    # if config.resume_from:
    #     model, optimizer, scheduler, checkpoint = load_checkpoint(
    #         config.resume_from, model, optimizer, scheduler, device
    #     )
    #     start_epoch = checkpoint["epoch"]
    #     start_batch = checkpoint.get("batch_idx", 0)
    #     best_loss = checkpoint.get("best_loss", float("inf"))
    #     logger.info(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    # Initial evaluation
    logger.info("Running initial evaluation...")
    eval_token_loss, eval_action_loss = evaluate_model(
        model,
        test_dataloader,
        device,
        epoch=0,
        global_step=0,
        wandb_logger=wandb_logger,
        save_dir=config.save_dir,
    )
    logger.info(
        f"Initial eval - Token Loss: {eval_token_loss:.6f}, Action Loss: {eval_action_loss:.6f}"
    )

    # Training loop
    logger.info("Starting training loop...")

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss, global_step = train_epoch(
                model,
                action_model,
                tokenizer,
                train_dataloader,
                test_dataloader,
                optimizer,
                scheduler,
                action_optimizer,
                action_scheduler,
                device,
                epoch,
                config,
                wandb_logger,
                epoch_start_batch,
                config.save_dir,
            )

            # End-of-epoch evaluation
            eval_token_loss, eval_action_loss = evaluate_model(
                model,
                test_dataloader,
                device,
                epoch,
                global_step,
                wandb_logger=wandb_logger,
                save_dir=config.save_dir,
            )
            eval_loss = eval_token_loss + eval_action_loss
            logger.info(
                f"End of epoch {epoch} eval - Token Loss: {eval_token_loss:.6f}, "
                f"Action Loss: {eval_action_loss:.6f}, Total: {eval_loss:.6f}"
            )

            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    len(train_dataloader),
                    avg_loss,
                    config,
                    best_loss,
                    train_dataloader.get_state(),
                    is_best=True,
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
            avg_loss,
            config,
            best_loss,
            dataloader_state,
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        if wandb_logger:
            wandb_logger.finish()

    logger.info("Training completed!")
    logger.info(f"Best loss achieved: {best_loss:.6f}")

    if config.use_s3 and s3_manager:
        logger.info(f"Checkpoints saved to S3 bucket: {s3_manager.bucket_name}")
    else:
        logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")

    if config.use_wandb:
        logger.info(f"Training metrics logged to Wandb project: {config.wandb_project}")


if __name__ == "__main__":
    config = tyro.cli(DynamicsModelTrainingConfig)
    logger.info(f"Starting training... config: {config.__dict__}")
    main(config)
    logger.info("Training completed!")
