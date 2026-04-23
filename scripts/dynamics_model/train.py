"""
python -m scripts.dynamics_model.train \
  --tokenizer_checkpoint_path fsq_tokenizer_2k_128_4_512_8_heads_4_layers/checkpoint_epoch1_batch2000.pt \
  --image_size 128 \
  --patch_size 4 \
  --num_images_in_video 5 \
  --batch_size 4 \
  --frames_dir pokemon_frames

  """
import logging
import math
import os
import time
from datetime import datetime
import traceback

import torch
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from data.data_loaders.factory import build_datasets, build_rollout_eval_dataset
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from data.s3.s3_utils import default_s3_manager
from dynamics_model.checkpoints import load_checkpoint, save_checkpoint
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.model import DynamicsModel
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from latent_action_model.model import LatentActionVQVAE
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.frechet_distance import compute_frechet_distance, compute_fvd
from monitoring.psnr import compute_delta_psnr, compute_frame_pixel_similarity
from monitoring.residual_coverage import compute_residual_coverage
from monitoring.setup_wandb import setup_wandb
from monitoring.videos import convert_video_to_images, save_comparison_images_next_frame, save_rollout_comparison_grid
from monitoring.wandb_media import log_image_batches
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def maskgit_predict_last_frame(
    model: DynamicsModel,
    video_batch: torch.Tensor,
    max_steps: int = 25,
) -> torch.Tensor:
    """Run MaskGIT inference to predict the last frame of ``video_batch``.

    ``model.inference`` now treats the last frame of its input as the masked
    target, so we pass the full ``video_batch`` (not ``video_batch[:, :-1]``)
    and supply the action describing the transition into the final frame.

    Args:
        video_batch: [B, T, C, H, W] full video including the target frame.
        max_steps: number of MaskGIT iterative-decoding steps.

    Returns:
        Predicted video of shape [B, T, C, H, W] with the last frame
        replaced by the model's prediction.
    """
    action_encoded = model.action_model.encode(video_batch)
    action_tokens = model.action_model.get_action_sequence(action_encoded)
    last_action = action_tokens[:, -1]  # (B,)
    return model.inference(video_batch, last_action, max_steps=max_steps)


def evaluate_model(
    model: DynamicsModel,
    dataloader: VideoWindowLoader,
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
    
    # Collect metrics for Δt PSNR computation
    psnr_inferred_list: list[float] = []
    psnr_random_list: list[float] = []
    delta_psnr_list: list[float] = []

    # Collect pixel similarity between last two frames (t-2 vs t-1)
    gt_next_frame_sim_list: list[float] = []
    pred_next_frame_sim_list: list[float] = []

    # Collect residual-coverage metrics (MaskGIT inference)
    residual_coverage_accum: list[dict[str, float]] = []

    # Collect action tokens across batches for codebook-usage metrics
    action_tokens_accum: list[torch.Tensor] = []

    eval_dir = f"{save_dir}/eval/epoch_{epoch}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving eval results to {eval_dir}")

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                video_batch = video_batch.to(device)
                decoded, token_loss, action_loss = model(video_batch)

                total_token_loss += token_loss.item() * video_batch.size(0)
                total_action_loss += action_loss.item() * video_batch.size(0)
                total_samples += video_batch.size(0)

                # Get reconstructed video from action model for FID/FVD
                # The action model reconstructs frames 1:T from frames 0:T-1
                action_encoded = model.action_model.encode(video_batch)
                action_tokens = model.action_model.get_action_sequence(action_encoded)
                reconstructed_video = model.action_model.decode(video_batch, action_encoded)

                # Real target frames: video[:, 1:, :, :, :]
                real_target_frames = video_batch
                real_frames_batches.append(real_target_frames.detach().cpu())
                pred_frames_batches.append(reconstructed_video.detach().cpu())

                # Track action-token usage for codebook metrics
                action_tokens_accum.append(action_tokens.detach().cpu())

                # Δt PSNR controllability metric (inferred vs. random actions)
                psnr_inferred, psnr_random, delta_psnr = compute_delta_psnr(
                    model.action_model, video_batch, device
                )
                if math.isfinite(psnr_inferred) and math.isfinite(psnr_random):
                    psnr_inferred_list.append(psnr_inferred)
                    psnr_random_list.append(psnr_random)
                    delta_psnr_list.append(delta_psnr)

                # Pixel similarity between the last two frames (t-2 vs t-1)
                # for both ground truth and predicted videos.
                if real_target_frames.shape[1] >= 2:
                    gt_next_frame_sim_list.append(
                        compute_frame_pixel_similarity(
                            real_target_frames[:, -2], real_target_frames[:, -1]
                        )
                    )
                if reconstructed_video.shape[1] >= 2:
                    pred_next_frame_sim_list.append(
                        compute_frame_pixel_similarity(
                            reconstructed_video[:, -2], reconstructed_video[:, -1]
                        )
                    )

                # Residual coverage via MaskGIT inference
                if video_batch.shape[1] >= 2:
                    pred_video_maskgit = maskgit_predict_last_frame(model, video_batch)
                    residual_coverage_accum.append(
                        compute_residual_coverage(
                            gt_frame=video_batch[:, -1],
                            pred_frame=pred_video_maskgit[:, -1],
                            prev_frame=video_batch[:, -2],
                        )
                    )

                # Save comparison images
                predicted_videos = convert_video_to_images(reconstructed_video)
                expected_videos = convert_video_to_images(real_target_frames)
                batch_dir = f"{eval_dir}/batch_{batch_idx}"
                os.makedirs(batch_dir, exist_ok=True)
                image_path = f"{batch_dir}/next_frame_comparison_grid.png"
                save_comparison_images_next_frame(
                    predicted_videos,
                    action_tokens.squeeze(-1).detach().cpu().numpy().tolist(),
                    expected_videos,
                    batch_dir,
                )

                # Log comparison image to wandb
                if wandb_logger:
                    wandb_logger.log(
                        {f"eval/comparison_{batch_idx}": wandb.Image(image_path)},
                        step=global_step,
                    )
                saved_image_paths.append(image_path)
                logger.debug(f"Saved comparison image to {image_path}")

            except Exception as e:
                traceback.print_exc()
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_token_loss = (
        total_token_loss / total_samples if total_samples > 0 else float("inf")
    )
    avg_action_loss = (
        total_action_loss / total_samples if total_samples > 0 else float("inf")
    )

    frechet_metrics = {}
    if real_frames_batches and pred_frames_batches:
        trimmed_frames = [frame[:, -2:, :, :, :] for frame in real_frames_batches]
        trimmed_pred_frames = [frame[:, -2:, :, :, :] for frame in pred_frames_batches]
        real_all = torch.cat(trimmed_frames, dim=0)
        pred_all = torch.cat(trimmed_pred_frames, dim=0)

        # Compute FID (Frechet Inception Distance) - frame-level metric
        logger.info(f"Computing FID between {real_all.shape} and {pred_all.shape}")
        t = time.time()
        frechet_distance = compute_frechet_distance(real_all, pred_all)
        logger.info(
            f"FID computed in {time.time() - t:.2f} seconds: {frechet_distance:.4f}"
        )
        # Compute FVD (Frechet Video Distance) - video-level metric
        logger.info(f"Computing FVD between {real_all.shape} and {pred_all.shape}")
        t = time.time()
        fvd_score = compute_fvd(real_all, pred_all)
        logger.info(
            f"FVD computed in {time.time() - t:.2f} seconds: {fvd_score:.4f}"
        )
        frechet_metrics = {
            "eval/fid": frechet_distance,
            "eval/fvd": fvd_score,
        }

    # Compute average PSNR metrics
    psnr_metrics = {}
    if psnr_inferred_list:
        avg_psnr_inferred = sum(psnr_inferred_list) / len(psnr_inferred_list)
        avg_psnr_random = sum(psnr_random_list) / len(psnr_random_list)
        avg_delta_psnr = sum(delta_psnr_list) / len(delta_psnr_list)
        psnr_metrics = {
            "eval/psnr_inferred": avg_psnr_inferred,
            "eval/psnr_random": avg_psnr_random,
            "eval/delta_psnr": avg_delta_psnr,
        }
        logger.info(
            f"PSNR metrics: inferred={avg_psnr_inferred:.4f}, random={avg_psnr_random:.4f}, "
            f"delta={avg_delta_psnr:.4f}"
        )

    # Pixel similarity between the last two frames (t-2 vs t-1).
    frame_sim_metrics: dict[str, float] = {}
    if gt_next_frame_sim_list:
        frame_sim_metrics["eval/ground_truth_next_frame_sim"] = sum(
            gt_next_frame_sim_list
        ) / len(gt_next_frame_sim_list)
    if pred_next_frame_sim_list:
        frame_sim_metrics["eval/predicted_next_frame_sim"] = sum(
            pred_next_frame_sim_list
        ) / len(pred_next_frame_sim_list)

    # Action codebook usage across all eval batches.
    codebook_metrics: dict[str, float] = {}
    if action_tokens_accum:
        all_action_tokens = torch.cat(
            [t.reshape(-1) for t in action_tokens_accum], dim=0
        )
        usage = compute_codebook_usage(
            all_action_tokens, model.action_model.action_vocab_size
        )
        codebook_metrics = {f"eval/action_codebook_{k}": v for k, v in usage.items()}
        logger.info(
            f"Action codebook usage: "
            f"unique={int(usage['num_unique'])}/{model.action_model.action_vocab_size} "
            f"({usage['usage_fraction']:.2%}), "
            f"perplexity={usage['perplexity']:.2f}, "
            f"norm_entropy={usage['normalized_entropy']:.4f}"
        )

    # Aggregate residual-coverage metrics across batches.
    residual_metrics: dict[str, float] = {}
    if residual_coverage_accum:
        keys = residual_coverage_accum[0].keys()
        residual_metrics = {
            f"eval/{k}": sum(d[k] for d in residual_coverage_accum) / len(residual_coverage_accum)
            for k in keys
        }
        logger.info(
            f"Residual coverage: R²={residual_metrics['eval/residual_r2']:.4f}, "
            f"cosine={residual_metrics['eval/residual_cosine']:.4f}, "
            f"changed_px_mse={residual_metrics['eval/changed_pixel_mse']:.6f}, "
            f"changed_frac={residual_metrics['eval/changed_pixel_fraction']:.4f}"
        )

    # Build log string with available metrics
    log_parts = [f"token_loss={avg_token_loss:.6f}", f"action_loss={avg_action_loss:.6f}"]
    if frechet_metrics:
        log_parts.append(f"FID={frechet_metrics.get('eval/fid', float('nan')):.4f}")
        log_parts.append(f"FVD={frechet_metrics.get('eval/fvd', float('nan')):.4f}")
    if psnr_metrics:
        log_parts.append(f"delta_PSNR={psnr_metrics.get('eval/delta_psnr', float('nan')):.4f}")
    if frame_sim_metrics:
        log_parts.append(
            f"gt_sim={frame_sim_metrics.get('eval/ground_truth_next_frame_sim', float('nan')):.4f}"
        )
        log_parts.append(
            f"pred_sim={frame_sim_metrics.get('eval/predicted_next_frame_sim', float('nan')):.4f}"
        )
    if residual_metrics:
        log_parts.append(f"residual_R²={residual_metrics.get('eval/residual_r2', float('nan')):.4f}")
    if codebook_metrics:
        log_parts.append(
            f"codebook_use={codebook_metrics.get('eval/action_codebook_usage_fraction', float('nan')):.2%}"
        )
        log_parts.append(
            f"codebook_ppl={codebook_metrics.get('eval/action_codebook_perplexity', float('nan')):.2f}"
        )
    log_parts.append(f"saved {len(saved_image_paths)} images")
    
    logger.info(f"Eval complete: {', '.join(log_parts)}")

    # Log to wandb if available
    if wandb_logger and global_step is not None:
        log_dict = {
            "eval/token_loss": avg_token_loss,
            "eval/action_loss": avg_action_loss,
            "eval/total_loss": avg_token_loss + avg_action_loss,
            "eval/epoch": epoch,
            **frechet_metrics,
            **psnr_metrics,
            **frame_sim_metrics,
            **residual_metrics,
            **codebook_metrics,
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


def evaluate_model_rollout(
    model: DynamicsModel,
    rollout_dataloader: VideoWindowLoader,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: DynamicsModelTrainingConfig,
    wandb_logger=None,
    save_dir: str = "dynamics_model_results",
) -> None:
    """Run teacher-forced and autoregressive rollout evals.

    Expects ``rollout_dataloader`` to yield clips of length
    ``2 * config.num_images_in_video`` (2T). For each clip the function:

    1. Extracts GT actions for the full 2T clip via the action model.
    2. Saves a teacher-forced comparison grid from ``model.inference`` over the
       entire 2T clip, with predictions beginning at frame ``T - 1``.
    3. Saves an autoregressive rollout grid from ``model.rollout`` seeded by
       the first T GT frames, with predictions beginning at frame ``T``.
    """
    model.eval()
    T = config.num_images_in_video
    saved_rollout_image_paths: list[str] = []
    saved_teacher_forced_image_paths: list[str] = []

    eval_dir = f"{save_dir}/eval_rollout/epoch_{epoch}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Running rollout eval ({config.rollout_eval_batches} batches) → {eval_dir}")

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(rollout_dataloader):
            if batch_idx >= config.rollout_eval_batches:
                break

            try:
                video_batch = video_batch.to(device)  # (B, 2T, C, H, W)

                # Extract GT actions for the full 2T clip
                action_encoded = model.action_model.encode(video_batch)
                actions_full = model.action_model.get_action_sequence(
                    action_encoded
                )  # (B, 2T-1)

                gt_images = convert_video_to_images(video_batch)

                teacher_forced_full = model.inference(
                    video_batch,
                    actions_full[:, -1],
                    max_steps=config.rollout_max_steps,
                )  # (B, 2T, C, H, W)
                teacher_forced_images = convert_video_to_images(teacher_forced_full)
                teacher_forced_actions = (
                    actions_full[:, T - 2 :].detach().cpu().numpy().tolist()
                )

                rollout_actions = (
                    actions_full[:, T - 1 :]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                predicted_full = model.rollout(
                    video_batch[:, :T],
                    actions_full[:, T - 1 :],
                    max_steps=config.rollout_max_steps,
                )  # (B, 2T, C, H, W)
                rollout_images = convert_video_to_images(predicted_full)

                batch_dir = f"{eval_dir}/batch_{batch_idx}"
                os.makedirs(batch_dir, exist_ok=True)
                save_rollout_comparison_grid(
                    gt_videos=gt_images,
                    predicted_videos=teacher_forced_images,
                    predicted_actions=teacher_forced_actions,
                    output_dir=batch_dir,
                    prediction_start_idx=T - 1,
                    file_suffix="teacher_forced_comparison_grid.png",
                )
                save_rollout_comparison_grid(
                    gt_videos=gt_images,
                    predicted_videos=rollout_images,
                    predicted_actions=rollout_actions,
                    output_dir=batch_dir,
                    prediction_start_idx=T,
                )

                rollout_image_path = f"{batch_dir}/rollout_comparison_grid.png"
                teacher_forced_image_path = (
                    f"{batch_dir}/teacher_forced_comparison_grid.png"
                )
                saved_rollout_image_paths.append(rollout_image_path)
                saved_teacher_forced_image_paths.append(teacher_forced_image_path)

                if wandb_logger:
                    wandb_logger.log(
                        {
                            f"eval_rollout/comparison_{batch_idx}": wandb.Image(
                                rollout_image_path
                            ),
                            f"eval_teacher_forced/comparison_{batch_idx}": wandb.Image(
                                teacher_forced_image_path
                            ),
                        },
                        step=global_step,
                    )

                logger.debug(
                    "Saved rollout comparison to "
                    f"{rollout_image_path} and teacher-forced comparison to "
                    f"{teacher_forced_image_path}"
                )

            except Exception as e:
                traceback.print_exc()
                logger.warning(f"Error in rollout eval batch {batch_idx}: {e}")
                continue

    logger.info(
        "Rollout eval complete: saved "
        f"{len(saved_rollout_image_paths)} rollout grids and "
        f"{len(saved_teacher_forced_image_paths)} teacher-forced grids"
    )

    if wandb_logger and saved_rollout_image_paths:
        log_image_batches(
            wandb_logger,
            key_prefix="eval_rollout/comparison",
            image_paths=saved_rollout_image_paths,
            batch_size=5,
            step=global_step,
        )
    if wandb_logger and saved_teacher_forced_image_paths:
        log_image_batches(
            wandb_logger,
            key_prefix="eval_teacher_forced/comparison",
            image_paths=saved_teacher_forced_image_paths,
            batch_size=5,
            step=global_step,
        )

    model.train()


def train_epoch(
    dynamics_model: DynamicsModel,
    action_model: LatentActionVQVAE,
    train_dataloader: VideoWindowLoader,
    test_dataloader: VideoWindowLoader,
    rollout_dataloader: VideoWindowLoader | None,
    dynamics_optimizer: optim.Optimizer,
    dynamics_scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: DynamicsModelTrainingConfig,
    wandb_logger,
    global_step: int,
    start_batch: int = 0,
    save_dir: str = "dynamics_model_results",
):
    """Train for one epoch"""
    total_loss = 0.0
    total_optimizer_step_loss = 0.0
    num_batches = len(train_dataloader)
    best_loss = float("inf")
    accumulation_steps = config.gradient_accumulation_steps

    # Set up resumable dataloader
    if start_batch > 0:
        train_dataloader.resumable_loader.set_epoch(epoch)
        train_dataloader.resumable_loader.current_batch = start_batch

    os.makedirs(f"{save_dir}/train/epoch_{epoch}", exist_ok=True)

    epoch_start_time = time.time()
    action_loss_acc, token_loss_acc = [], []
    dynamics_optimizer.zero_grad()

    for batch_idx, video_batch in enumerate(train_dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Move to device
        video_batch = video_batch.to(device)

        # Forward pass - MaskGIT returns (predictions, token_loss, action_loss)
        decoded, token_loss, action_loss = dynamics_model(video_batch)

        # Scale each microbatch by the size of its accumulation window so the
        # accumulated gradient matches the corresponding large-batch mean.
        remaining_batches = num_batches - batch_idx
        current_window_size = min(accumulation_steps, remaining_batches)
        combined_loss = (token_loss + action_loss) / current_window_size
        combined_loss.backward()
        token_loss_acc.append(token_loss.item())
        action_loss_acc.append(action_loss.item())

        # Only step optimizer after accumulating enough gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            # Gradient clipping
            if config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    dynamics_model.parameters(), max_norm=config.gradient_clipping
                )

            dynamics_optimizer.step()
            dynamics_scheduler.step()
            dynamics_optimizer.zero_grad()

            avg_token_loss = sum(token_loss_acc) / len(token_loss_acc)
            avg_action_loss = sum(action_loss_acc) / len(action_loss_acc)
            total_loss = avg_token_loss + avg_action_loss
            total_optimizer_step_loss += total_loss

            batch_time = time.time() - batch_start_time

            # Log to wandb
            if wandb_logger:
                current_lrs = dynamics_scheduler.get_last_lr()
                wandb_metrics = {
                    "train/loss": total_loss,
                    "train/token_loss": avg_token_loss,
                    "train/action_loss": avg_action_loss,
                    "train/learning_rate": current_lrs[0],
                    "train/dynamics_learning_rate": current_lrs[0],
                    "train/action_learning_rate": current_lrs[1],
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                }
                wandb_logger.log(wandb_metrics, step=global_step)

            # Log progress
            if (batch_idx // accumulation_steps) % config.log_interval == 0:
                current_lrs = dynamics_scheduler.get_last_lr()
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {total_loss:.6f} (token: {avg_token_loss:.6f}, action: {avg_action_loss:.6f}), "
                    f"LRs: dynamics={current_lrs[0]:.2e}, action={current_lrs[1]:.2e}, "
                    f"Time: {batch_time:.2f}s"
                )

            action_loss_acc, token_loss_acc = [], []

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

                if rollout_dataloader is not None:
                    evaluate_model_rollout(
                        dynamics_model,
                        rollout_dataloader,
                        device,
                        epoch,
                        global_step,
                        config,
                        wandb_logger=wandb_logger,
                        save_dir=config.save_dir,
                    )

                # Residual coverage on the current train batch
                if video_batch.shape[1] >= 2:
                    dynamics_model.eval()
                    train_coverage = compute_residual_coverage(
                        gt_frame=video_batch[:, -1],
                        pred_frame=maskgit_predict_last_frame(dynamics_model, video_batch)[:, -1],
                        prev_frame=video_batch[:, -2],
                    )
                    dynamics_model.train()
                    logger.info(
                        f"Train residual coverage: R²={train_coverage['residual_r2']:.4f}, "
                        f"cosine={train_coverage['residual_cosine']:.4f}"
                    )
                    if wandb_logger:
                        wandb_logger.log(
                            {f"train/{k}": v for k, v in train_coverage.items()},
                            step=global_step,
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
                    action_model=action_model,
                )

            # Increment global step
            global_step += 1

    # Calculate average loss over optimizer steps
    num_batches_processed = max(num_batches - start_batch, 0)
    num_optimizer_steps = math.ceil(num_batches_processed / accumulation_steps)
    avg_loss = total_optimizer_step_loss / max(num_optimizer_steps, 1)
    epoch_time = time.time() - epoch_start_time

    # Log epoch summary
    if wandb_logger:
        epoch_metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_time": epoch_time,
            "train/epoch": epoch,
        }
        wandb_logger.log(epoch_metrics, step=global_step)

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

    train_dataset, test_dataset = build_datasets(config, local_cache)

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

    # Build a 2T-frame eval dataset for rollout evaluation
    rollout_window = 2 * config.num_images_in_video
    logger.info(f"Building rollout eval dataset with window={rollout_window}...")
    try:
        rollout_dataset = build_rollout_eval_dataset(config, local_cache, rollout_window)
        rollout_dataloader: VideoWindowLoader | None = VideoWindowLoader(
            dataset=rollout_dataset,
            batch_size=config.batch_size,
            image_size=config.image_size,
            shuffle=True,
            num_workers=4,
            seed=config.seed,
        )
        logger.info(f"Rollout eval dataset: {len(rollout_dataset)} samples")
    except Exception as e:
        logger.warning(f"Could not build rollout eval dataset (skipping rollout eval): {e}")
        rollout_dataloader = None

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
    action_params = [
        parameter for parameter in action_model.parameters() if parameter.requires_grad
    ]
    action_param_ids = {id(parameter) for parameter in action_params}
    dynamics_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in action_param_ids
    ]
    optimizer = optim.AdamW(
        [
            {"params": dynamics_params, "lr": config.dynamics_learning_rate},
            {"params": action_params, "lr": config.action_learning_rate},
        ],
        weight_decay=1e-4,
    )
    logger.info(
        "Optimizer created with parameter groups: "
        f"dynamics_lr={config.dynamics_learning_rate}, "
        f"action_lr={config.action_learning_rate}"
    )

    # Cosine annealing scheduler with warmup
    steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_steps

    # Linear warmup from 0 to learning_rate over warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
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

    if config.dynamics_model_checkpoint_path:
        model, optimizer, scheduler, checkpoint = load_checkpoint(
            config.dynamics_model_checkpoint_path, model, optimizer, scheduler, device
        )
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint.get("batch_idx", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))

        # End-of-epoch checkpoints store the last seen batch index, so advance to
        # the next epoch instead of re-entering a completed one.
        if start_batch >= len(train_dataloader):
            start_epoch += 1
            start_batch = 0

        logger.info(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    # Initial evaluation
    logger.info("Running initial evaluation...")
    global_step = 0
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

    # Skip rollout evaluation before any optimizer steps have run.
    if rollout_dataloader is not None and global_step > 0:
        evaluate_model_rollout(
            model,
            rollout_dataloader,
            device,
            epoch=0,
            global_step=0,
            config=config,
            wandb_logger=wandb_logger,
            save_dir=config.save_dir,
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
                train_dataloader,
                test_dataloader,
                rollout_dataloader,
                optimizer,
                scheduler,
                device,
                epoch,
                config,
                wandb_logger,
                global_step,
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

            if rollout_dataloader is not None:
                evaluate_model_rollout(
                    model,
                    rollout_dataloader,
                    device,
                    epoch,
                    global_step,
                    config,
                    wandb_logger=wandb_logger,
                    save_dir=config.save_dir,
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
                    action_model=action_model,
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
            action_model=action_model,
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
