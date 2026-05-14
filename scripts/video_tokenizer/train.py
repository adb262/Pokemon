# python -m scripts.video_tokenizer.train --frames_dir pokemon_frames/pokemon_emerald --num_images_in_video 5 --batch_size 16 --save_dir fsq_tokenizer_2k_128_4_512_8_heads_4_layers --bins 8 8 6 5 --use_s3 --dataset_train_key pokemon_emerald_train_0_9_5_frames.json --checkpoint_dir fsq_tokenizer_2k_128_4_512_8_heads_4_layers --patch_size 4 --image_size 128 --d_model 512 --num_heads 8 --num_transformer_layers 4 --num_epochs 20
# from beartype import BeartypeConf
# from beartype.claw import beartype_all
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import torch
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from data.s3.s3_utils import default_s3_manager
from loss.loss_fns import clipped_l2_reconstruction_loss, reconstruction_loss
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.experiment_logger import ExperimentLogger, resolve_logging_backend
from monitoring.videos import convert_video_to_images, save_comparison_images
from scripts.video_tokenizer.eval import eval_model
from torch_utilities.initialize import init_weights
from video_tokenization.checkpoints import save_checkpoint
from video_tokenization.create_tokenizer import create_model
from video_tokenization.model import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    """Track eval-loss improvement at evaluation granularity.

    The counter increments once per evaluation event (whether the eval ran
    mid-epoch via ``eval_interval`` or at an epoch boundary) and resets on
    any improvement strictly greater than ``min_delta``. ``patience`` is
    therefore measured in evaluations, not epochs, so users can stop on
    sub-epoch staleness when ``eval_interval`` divides epochs into many
    evals.
    """

    best_loss: float = float("inf")
    evals_without_improvement: int = 0

    def update(self, eval_loss: float, min_delta: float) -> bool:
        """Record an eval result and return ``True`` iff it improved the best."""
        if eval_loss < self.best_loss - min_delta:
            self.best_loss = eval_loss
            self.evals_without_improvement = 0
            return True
        self.evals_without_improvement += 1
        return False

    def should_stop(self, patience: int) -> bool:
        """``patience <= 0`` disables early stopping entirely."""
        return patience > 0 and self.evals_without_improvement >= patience


def build_reconstruction_criterion(
    config: VideoTokenizerTrainingConfig,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the reconstruction loss callable selected by ``config``.

    The returned callable matches the ``criterion(video, decoded)`` shape used
    throughout the trainer/eval. ``clipped_l2`` floors the per-pixel MSE at
    ``l2_clip_c / 255**2`` so trivial pixels can't drive the loss to zero,
    mirroring the action-decoder ``clipped_l2`` option in the dynamics model.
    """
    if config.reconstruction_loss_type == "l2":
        return reconstruction_loss

    if config.reconstruction_loss_type == "clipped_l2":
        l2_clip_c = config.l2_clip_c

        def clipped_criterion(
            video: torch.Tensor, decoded: torch.Tensor
        ) -> torch.Tensor:
            return clipped_l2_reconstruction_loss(
                decoded,
                video,
                l2_clip_c=l2_clip_c,
            )

        return clipped_criterion
    raise ValueError(
        f"Unknown reconstruction_loss_type: {config.reconstruction_loss_type!r}"
    )


def train_epoch(
    model: VideoTokenizer,
    dataloader: VideoWindowLoader,
    test_dataloader: VideoWindowLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    config: VideoTokenizerTrainingConfig,
    experiment_logger: ExperimentLogger,
    early_stopping_state: EarlyStoppingState,
    start_batch: int = 0,
    save_dir: str = "tokenization_results",
):
    total_loss = 0.0
    num_batches = len(dataloader)
    accumulation_steps = config.gradient_accumulation_steps
    should_early_stop = False

    # Configure the resumable dataloader: set the per-epoch shuffle and skip
    # ahead to ``start_batch`` at the sampler level, so workers never load
    # (and discard) the skipped batches.
    dataloader.resumable_loader.set_epoch(epoch)
    dataloader.resumable_loader.set_start_batch(start_batch)

    epoch_start_time = time.time()
    batch_start_time = time.time()
    accumulated_loss = 0.0
    logged_codebook_tokens: list[torch.Tensor] = []
    saved_on_last_step = False

    # ``enumerate(..., start=start_batch)`` keeps ``batch_idx`` aligned with
    # the absolute position in the (full) epoch, so gradient accumulation,
    # logging, and end-of-epoch detection all match a fresh-start run.
    for batch_idx, video_batch in enumerate(dataloader, start=start_batch):
        batch_start_time = time.time()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass with decoder.
        # Outputs a tensor of shape (batch_size, num_images_in_video, channels, image_height, image_width).
        quantized = model.encode(video_batch)
        decoded = model.decode(quantized)
        codebook_tokens = model.quantized_value_to_codes(quantized.detach())
        logged_codebook_tokens.append(codebook_tokens.detach().cpu())

        # Calculate loss (reconstruction loss)
        # Scale loss by accumulation steps for correct gradient averaging
        loss = criterion(video_batch, decoded) / accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Track unscaled loss for logging
        accumulated_loss += loss.item() * accumulation_steps

        # Only step optimizer after accumulating enough gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Step scheduler every optimizer step for cosine annealing
            optimizer.zero_grad()
            saved_on_last_step = False

            # Calculate average loss over accumulated steps
            avg_accumulated_loss = accumulated_loss / min(
                accumulation_steps, (batch_idx % accumulation_steps) + 1
            )
            total_loss += avg_accumulated_loss
            codebook_token_window = (
                torch.cat([tokens.reshape(-1) for tokens in logged_codebook_tokens], dim=0)
                if logged_codebook_tokens
                else torch.empty(0, dtype=torch.long)
            )
            codebook_usage = compute_codebook_usage(
                codebook_token_window, model.get_vocab_size()
            )
            batch_time = time.time() - batch_start_time

            # Calculate global step (counts optimizer steps, not batches)
            global_step = epoch * (num_batches // accumulation_steps) + (
                batch_idx // accumulation_steps
            )

            # Log training metrics with system metrics.
            training_metrics = {
                "train/loss": avg_accumulated_loss,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/batch_time": batch_time,
                "train/epoch": epoch,
                "train/batch": batch_idx,
            }
            training_metrics.update(
                {
                    f"train/codebook_{key}": value
                    for key, value in codebook_usage.items()
                }
            )

            experiment_logger.log(training_metrics, step=global_step)

            # Log progress
            if (batch_idx // accumulation_steps) % config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {avg_accumulated_loss:.6f}, LR: {current_lr:.2e}, "
                    f"Codebook: unique={int(codebook_usage['num_unique'])}/{model.get_vocab_size()} "
                    f"({codebook_usage['usage_fraction']:.2%}), "
                    f"ppl={codebook_usage['perplexity']:.2f}, "
                    f"Time: {batch_time:.2f}s"
                )

            # Reset accumulated loss
            accumulated_loss = 0.0
            logged_codebook_tokens = []

            # Save checkpoint + eval periodically (based on cumulative optimizer steps)
            if global_step > 0 and global_step % config.eval_interval == 0:
                dataloader_state = dataloader.get_state()
                saved_on_last_step = True

                # Save one fresh train comparison_grid per eval step. The
                # ``max_comparison_images`` cap applies *within* a single
                # save step (eval iterates multiple batches; train only has
                # the current batch), so we always emit one image here and
                # never accumulate across an epoch.
                predicted_videos = convert_video_to_images(decoded)
                expected_videos = convert_video_to_images(video_batch)
                comparison_path = f"{save_dir}/train/epoch_{epoch}/batch_{batch_idx}/comparison_grid.png"
                save_comparison_images(
                    predicted_videos, expected_videos, comparison_path
                )

                experiment_logger.log_image_batches(
                    key_prefix="train/comparison",
                    image_paths=[comparison_path],
                    batch_size=config.max_comparison_images,
                    step=global_step,
                )

                eval_loss = eval_model(
                    model,
                    test_dataloader,
                    criterion,
                    device,
                    epoch,
                    wandb_logger=experiment_logger,
                    save_dir=config.save_dir,
                    global_step=global_step,
                    max_comparison_images=config.max_comparison_images,
                )

                improved = early_stopping_state.update(
                    eval_loss, config.early_stopping_min_delta
                )
                if improved:
                    logger.info(
                        f"New best eval loss: {early_stopping_state.best_loss:.6f} "
                        f"(step {global_step})"
                    )
                elif config.early_stopping_patience > 0:
                    remaining = (
                        config.early_stopping_patience
                        - early_stopping_state.evals_without_improvement
                    )
                    logger.info(
                        f"No eval improvement for "
                        f"{early_stopping_state.evals_without_improvement} eval(s) "
                        f"(stopping in {max(remaining, 0)} more)"
                    )

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    batch_idx,
                    avg_accumulated_loss,
                    config,
                    early_stopping_state.best_loss,
                    dataloader_state,
                )

                if early_stopping_state.should_stop(config.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered at step {global_step} "
                        f"(best eval loss: {early_stopping_state.best_loss:.6f})"
                    )
                    if experiment_logger:
                        experiment_logger.log(
                            {
                                "train/early_stopped": 1,
                                "train/early_stopped_step": global_step,
                                "train/early_stopped_epoch": epoch,
                            },
                            step=global_step,
                        )
                    should_early_stop = True
                    break

    # Calculate average loss over optimizer steps
    num_optimizer_steps = num_batches // accumulation_steps
    avg_loss = total_loss / max(num_optimizer_steps, 1)
    epoch_time = time.time() - epoch_start_time

    # Log epoch metrics.
    if experiment_logger:
        epoch_metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_time": epoch_time,
            "train/epoch": epoch,
        }

        experiment_logger.log(epoch_metrics, step=epoch * num_batches)

    logger.info(
        f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s"
    )
    return avg_loss, global_step, saved_on_last_step, should_early_stop


def main(config: VideoTokenizerTrainingConfig):
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
        group="video-tokenizer-test",
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

    train_dataset, test_dataset = build_datasets(
        config, local_cache, test_limit=config.test_dataset_limit
    )

    logger.info(f"Creating data loader with {len(train_dataset)} videos...")
    train_dataloader = VideoWindowLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
    )
    test_dataloader = VideoWindowLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=8,
        seed=config.seed,
    )

    # Print dataset info
    train_info = train_dataloader.get_dataset_info()
    test_info = test_dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in train_info.items():
        logger.info(f"  {key}: {value}")

    for key, value in test_info.items():
        logger.info(f"  {key}: {value}")

    # Log dataset info without advancing the step counter when supported.
    if experiment_logger:
        experiment_logger.log(
            {f"dataset/{key}": value for key, value in train_info.items()},
            commit=False,
        )
        experiment_logger.log(
            {f"test_dataset/{key}": value for key, value in test_info.items()},
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
    model = create_model(config)
    model.to(device)

    logger.info(
        f"Num params: {sum(p.numel() for p in model.parameters())} on device {device}"
    )

    if experiment_logger:
        experiment_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    logger.info(f"Optimizer created with learning rate: {config.learning_rate}")

    # Cosine annealing scheduler with warmup (steps per epoch is reduced by accumulation)
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_steps

    # Linear warmup from 0 to learning_rate over warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,  # Start from near-zero
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Cosine annealing for the remaining steps
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.min_learning_rate,
    )

    # Combine warmup + cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    logger.info(f"Warmup steps: {warmup_steps}")

    # Log effective batch size
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total optimizer steps: {total_steps}")

    criterion = build_reconstruction_criterion(config)
    logger.info(
        f"Reconstruction loss: {config.reconstruction_loss_type}"
        + (
            f" (l2_clip_c={config.l2_clip_c})"
            if config.reconstruction_loss_type == "clipped_l2"
            else ""
        )
    )
    s3_manager = default_s3_manager

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0
    # Training loop
    logger.info("Starting training loop...")
    early_stopping_state = EarlyStoppingState()
    if config.early_stopping_patience > 0:
        logger.info(
            f"Early stopping enabled: patience={config.early_stopping_patience} "
            f"eval(s), min_delta={config.early_stopping_min_delta}"
        )

    # Apply our init scheme before any eval so the pre-training eval reflects
    # the same initialization used during training.
    model.apply(init_weights)

    # first, evaluate on test dataset
    test_loss = eval_model(
        model,
        test_dataloader,
        criterion,
        device,
        epoch=0,
        wandb_logger=experiment_logger,
        save_dir=config.save_dir,
        global_step=0,
        max_comparison_images=config.max_comparison_images,
    )
    logger.info(f"Test loss: {test_loss:.6f}")

    try:
        model.train()
        avg_loss = float("inf")
        global_step = 0
        epoch = start_epoch
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss, global_step, saved_on_last_step, should_early_stop = train_epoch(
                model,
                train_dataloader,
                test_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                config,
                experiment_logger,
                early_stopping_state,
                epoch_start_batch,
                config.save_dir,
            )
            if should_early_stop:
                break

            # Skip the post-epoch eval when the periodic eval already ran on
            # the final optimizer step of this epoch — otherwise we'd
            # double-count an evaluation against the patience counter.
            if saved_on_last_step:
                logger.info(
                    f"Skipping post-epoch eval at step {global_step}: "
                    "periodic eval already ran on the last optimizer step."
                )
                model.train()
                continue

            eval_loss = eval_model(
                model,
                test_dataloader,
                criterion,
                device,
                epoch,
                wandb_logger=experiment_logger,
                save_dir=config.save_dir,
                global_step=global_step,
                max_comparison_images=config.max_comparison_images,
            )
            logger.info(f"Epoch {epoch} eval loss: {eval_loss:.6f}")

            improved = early_stopping_state.update(
                eval_loss, config.early_stopping_min_delta
            )
            if improved:
                logger.info(
                    f"New best eval loss: {early_stopping_state.best_loss:.6f} "
                    f"(epoch {epoch})"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    len(train_dataloader),
                    avg_loss,
                    config,
                    early_stopping_state.best_loss,
                    train_dataloader.get_state(),
                )
            elif config.early_stopping_patience > 0:
                remaining = (
                    config.early_stopping_patience
                    - early_stopping_state.evals_without_improvement
                )
                logger.info(
                    f"No eval improvement for "
                    f"{early_stopping_state.evals_without_improvement} eval(s) "
                    f"(stopping in {max(remaining, 0)} more)"
                )

            if early_stopping_state.should_stop(config.early_stopping_patience):
                logger.info(
                    f"Early stopping triggered after epoch {epoch} "
                    f"(best eval loss: {early_stopping_state.best_loss:.6f})"
                )
                if experiment_logger:
                    experiment_logger.log(
                        {
                            "train/early_stopped": 1,
                            "train/early_stopped_step": global_step,
                            "train/early_stopped_epoch": epoch,
                        },
                        step=global_step,
                    )
                break

            model.train()

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
            early_stopping_state.best_loss,
            dataloader_state,
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
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
    logger.info(f"Best loss achieved: {early_stopping_state.best_loss:.6f}")

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
    elif (
        logging_backend == "tensorboard"
        and experiment_logger.tensorboard_log_dir is not None
    ):
        logger.info(
            f"Training metrics logged to TensorBoard: {experiment_logger.tensorboard_log_dir}"
        )


if __name__ == "__main__":
    config = tyro.cli(VideoTokenizerTrainingConfig)

    logger.info(f"Starting training... config: {config.__dict__}")
    main(config)
    logger.info("Training completed!")
