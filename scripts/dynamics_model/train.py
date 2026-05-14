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
import torch._dynamo

import torch
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from data.s3.s3_utils import default_s3_manager
from dynamics_model.checkpoints import load_checkpoint, save_checkpoint
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.model import DynamicsModel
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from latent_action_model.model import LatentActionVQVAE
from monitoring.action_code_counts import format_top_code_counts, get_top_code_counts
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.experiment_logger import ExperimentLogger, resolve_logging_backend
from monitoring.frechet_distance import compute_frechet_distance, compute_fvd
from monitoring.psnr import compute_delta_psnr, compute_frame_pixel_similarity, compute_psnr
from monitoring.residual_coverage import compute_residual_coverage
from monitoring.videos import (
    convert_video_to_images,
    save_comparison_images_next_frame,
    save_residual_comparison_images,
    save_rollout_comparison_grid,
)
from schedulers.inverse_sigmoid_decay import inverse_sigmoid_decay
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer

torch._dynamo.config.cache_size_limit = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

VISUALIZATION_SEQUENCE_FRAMES = 5
MAX_EVAL_SAVED_IMAGES = 5


def _prediction_boundary_window(
    num_frames: int,
    prediction_start_idx: int,
) -> tuple[slice, int]:
    """Return a short frame window that includes the prediction boundary."""
    if num_frames <= VISUALIZATION_SEQUENCE_FRAMES:
        return slice(0, num_frames), prediction_start_idx

    start = min(
        max(prediction_start_idx - 1, 0),
        num_frames - VISUALIZATION_SEQUENCE_FRAMES,
    )
    end = start + VISUALIZATION_SEQUENCE_FRAMES
    return slice(start, end), prediction_start_idx - start


def _slice_actions_for_rollout_grid(
    actions: torch.Tensor,
    full_prediction_start_idx: int,
    frame_window: slice,
) -> torch.Tensor:
    frame_start = 0 if frame_window.start is None else frame_window.start
    frame_stop = frame_start if frame_window.stop is None else frame_window.stop
    action_start = max(frame_start - full_prediction_start_idx, 0)
    action_end = max(frame_stop - full_prediction_start_idx, 0)
    return actions[:, action_start:action_end]


def _remaining_visualization_slots(saved_count: int) -> int:
    return max(MAX_EVAL_SAVED_IMAGES - saved_count, 0)


@torch.no_grad()
def reconstruct_predicted_frames_from_residuals(
    video_batch: torch.Tensor, predicted_residuals: torch.Tensor
) -> torch.Tensor:
    """Convert residual predictions [B, T-1, C, H, W] into next-frame predictions."""
    return torch.clamp(
        video_batch[:, :-1, :, :, :] + predicted_residuals,
        0.0,
        1.0,
    )


@torch.no_grad()
def compute_rollout_metrics(
    pred_frames: torch.Tensor,
    real_frames: torch.Tensor,
    prev_frames: torch.Tensor,
) -> dict[str, float]:
    """Compute per-step and aggregated rollout-quality metrics.

    Each input is a video of predicted frames ``[B, K, C, H, W]``.
    For each step ``k`` in ``[0, K)``:

    * ``pred_frames[:, k]`` is the predicted frame at horizon ``k``.
    * ``real_frames[:, k]`` is the ground-truth frame at horizon ``k``.
    * ``prev_frames[:, k]`` is the ground-truth frame *immediately before*
      ``real_frames[:, k]`` (used as the copy-prev baseline for residual R²).

    Returns a dict containing:

    * ``per_step_<metric>/step_<k>`` - the value of ``<metric>`` at step ``k``.
    * ``mean_<metric>`` - mean of ``<metric>`` across all ``K`` steps.
    * ``final_<metric>`` - value at the most-drifted step (``k = K - 1``).

    where ``<metric>`` ranges over the residual-coverage metrics
    (``residual_r2``, ``residual_cosine``, ``changed_pixel_mse``, ``pred_mse``,
    ``copy_prev_mse``, ``changed_pixel_fraction``) plus ``psnr``.
    """
    if pred_frames.shape != real_frames.shape:
        raise ValueError(
            f"pred_frames and real_frames must match: "
            f"{pred_frames.shape} vs {real_frames.shape}"
        )
    if pred_frames.shape != prev_frames.shape:
        raise ValueError(
            f"pred_frames and prev_frames must match: "
            f"{pred_frames.shape} vs {prev_frames.shape}"
        )
    if pred_frames.dim() != 5:
        raise ValueError(
            f"Expected 5D tensors [B, K, C, H, W], got shape {pred_frames.shape}"
        )

    num_steps = pred_frames.shape[1]
    per_step_metrics: list[dict[str, float]] = []
    for k in range(num_steps):
        coverage = compute_residual_coverage(
            gt_frame=real_frames[:, k],
            pred_frame=pred_frames[:, k],
            prev_frame=prev_frames[:, k],
        )
        coverage["psnr"] = compute_psnr(real_frames[:, k], pred_frames[:, k])
        per_step_metrics.append(coverage)

    metrics: dict[str, float] = {}
    metric_keys = list(per_step_metrics[0].keys())

    for k, step_metrics in enumerate(per_step_metrics):
        for key, val in step_metrics.items():
            metrics[f"per_step_{key}/step_{k}"] = val

    for key in metric_keys:
        finite_values = [step[key] for step in per_step_metrics if math.isfinite(step[key])]
        if finite_values:
            metrics[f"mean_{key}"] = sum(finite_values) / len(finite_values)
        else:
            metrics[f"mean_{key}"] = float("nan")

    for key, val in per_step_metrics[-1].items():
        metrics[f"final_{key}"] = val

    return metrics


def _aggregate_rollout_metrics(
    pred_batches: list[torch.Tensor],
    real_batches: list[torch.Tensor],
    prev_batches: list[torch.Tensor],
    prefix: str,
    label: str,
) -> dict[str, float]:
    """Concatenate per-batch frame slices and compute rollout metrics.

    Returns a flat ``{prefix}/<metric>`` dict suitable for logging. Includes
    per-step / mean / final residual-coverage and PSNR metrics plus FID and
    FVD computed once over the full prediction horizon.
    """
    if not pred_batches:
        return {}

    pred_all = torch.cat(pred_batches, dim=0)
    real_all = torch.cat(real_batches, dim=0)
    prev_all = torch.cat(prev_batches, dim=0)

    metrics = compute_rollout_metrics(pred_all, real_all, prev_all)

    logger.info(
        f"Computing {label} FID over {real_all.shape} vs {pred_all.shape}"
    )
    t = time.time()
    fid_score = compute_frechet_distance(real_all, pred_all)
    logger.info(
        f"{label} FID computed in {time.time() - t:.2f}s: {fid_score:.4f}"
    )
    logger.info(
        f"Computing {label} FVD over {real_all.shape} vs {pred_all.shape}"
    )
    t = time.time()
    fvd_score = compute_fvd(real_all, pred_all)
    logger.info(
        f"{label} FVD computed in {time.time() - t:.2f}s: {fvd_score:.4f}"
    )
    metrics["fid"] = fid_score
    metrics["fvd"] = fvd_score

    logger.info(
        f"{label} metrics: "
        f"mean R²={metrics.get('mean_residual_r2', float('nan')):.4f}, "
        f"final R²={metrics.get('final_residual_r2', float('nan')):.4f}, "
        f"mean MSE={metrics.get('mean_pred_mse', float('nan')):.6f}, "
        f"final MSE={metrics.get('final_pred_mse', float('nan')):.6f}, "
        f"mean PSNR={metrics.get('mean_psnr', float('nan')):.4f}, "
        f"final PSNR={metrics.get('final_psnr', float('nan')):.4f}, "
        f"FID={fid_score:.4f}, FVD={fvd_score:.4f}"
    )

    return {f"{prefix}/{key}": value for key, value in metrics.items()}


@torch.no_grad()
def maskgit_predict_last_frame(
    model: DynamicsModel,
    video_batch: torch.Tensor,
    max_steps: int = 25,
) -> torch.Tensor:
    """Roll out one step to predict the held-out last frame of ``video_batch``.

    The full ``video_batch`` is used only to infer the GT action token for the
    transition into the target frame. Generation is seeded with
    ``video_batch[:, :-1]`` so the target pixels are held out.

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
    return model.rollout(
        video_batch[:, :-1],
        last_action.unsqueeze(1),
        max_steps=max_steps,
    )


def evaluate_model(
    model: DynamicsModel,
    dataloader: VideoWindowLoader,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: DynamicsModelTrainingConfig,
    experiment_logger: ExperimentLogger | None = None,
    save_dir: str = "dynamics_model_results",
    num_batches: int = 10,
    split: str = "eval",
    num_frames: int | None = None,
) -> tuple[float, float]:
    """Evaluate model on a subset of data with FID/FVD metrics and comparison images."""
    if split not in ("eval", "train_eval"):
        raise ValueError(f"split must be 'eval' or 'train_eval', got {split!r}")

    model.eval()
    predict_action_residuals = model.predict_action_residuals
    total_token_loss = 0.0
    total_action_loss = 0.0
    total_samples = 0
    saved_residual_image_paths: list[str] = []
    saved_reconstructed_image_paths: list[str] = []
    saved_visualization_samples = 0

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

    eval_dir = f"{save_dir}/{split}/epoch_{epoch}/step_{global_step}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving {split} eval results to {eval_dir}")

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                video_batch = video_batch.to(device, non_blocking=True)
                if num_frames is not None:
                    video_batch = video_batch[:, :num_frames]

                # Teacher-forced token loss: encode the full clip once with
                # the action model, then feed the GT video in as both the
                # target video and the prediction basis so ``forward`` masks a
                # random subset of tokens and scores reconstruction of those
                # positions under ground-truth conditioning. This mirrors the
                # training forward-pass shape conventions (T-frame video +
                # T-1 action tokens) without running the autoregressive
                # rollout used during training.
                action_encoded = model.action_model.encode(video_batch)
                action_tokens = model.action_model.get_action_sequence(
                    action_encoded
                )
                decoded, token_loss, _ = model(
                    video_batch,
                    video_batch,
                    action_tokens,
                )
                action_decoded = model.action_model.decode(
                    video_batch, action_encoded
                )
                action_loss = model.action_loss_fn(video_batch, action_decoded)

                total_token_loss += token_loss.item() * video_batch.size(0)
                total_action_loss += action_loss.item() * video_batch.size(0)
                total_samples += video_batch.size(0)

                if predict_action_residuals:
                    reconstructed_residuals = action_decoded
                    reconstructed_next_frames = (
                        reconstruct_predicted_frames_from_residuals(
                            video_batch, reconstructed_residuals
                        )
                    )
                else:
                    reconstructed_residuals = None
                    reconstructed_next_frames = action_decoded

                # Real target frames correspond to the action-model predictions for frames 1:T.
                real_target_frames = video_batch[:, 1:, :, :, :]
                real_frames_batches.append(real_target_frames.detach().cpu())
                pred_frames_batches.append(reconstructed_next_frames.detach().cpu())

                # Track action-token usage for codebook metrics
                action_tokens_accum.append(action_tokens.detach().cpu())

                # Δt PSNR controllability metric (inferred vs. random actions)
                psnr_inferred, psnr_random, delta_psnr = compute_delta_psnr(
                    model,
                    video_batch,
                    device,
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
                if reconstructed_next_frames.shape[1] >= 2:
                    pred_next_frame_sim_list.append(
                        compute_frame_pixel_similarity(
                            reconstructed_next_frames[:, -2],
                            reconstructed_next_frames[:, -1],
                        )
                    )

                # Residual coverage via MaskGIT inference
                if video_batch.shape[1] >= 2:
                    pred_video_maskgit = maskgit_predict_last_frame(model, video_batch, config.rollout_max_denoising_steps)
                    residual_coverage_accum.append(
                        compute_residual_coverage(
                            gt_frame=video_batch[:, -1],
                            pred_frame=pred_video_maskgit[:, -1],
                            prev_frame=video_batch[:, -2],
                        )
                    )

                # Save comparison images for the reconstructed next frames, plus
                # a residual-comparison grid when the action model is configured
                # to predict residuals.
                vis_frames = min(VISUALIZATION_SEQUENCE_FRAMES, video_batch.shape[1])
                batch_dir = f"{eval_dir}/batch_{batch_idx}"
                image_path: str | None = None
                residual_image_path: str | None = None
                visualization_slots = _remaining_visualization_slots(
                    saved_visualization_samples
                )
                num_visualized_samples = min(video_batch.shape[0], visualization_slots)
                if vis_frames >= 2 and num_visualized_samples > 0:
                    os.makedirs(batch_dir, exist_ok=True)
                    vis_transition_count = vis_frames - 1
                    predicted_videos = convert_video_to_images(
                        reconstructed_next_frames[
                            :num_visualized_samples, :vis_transition_count
                        ]
                    )
                    expected_videos = convert_video_to_images(
                        video_batch[:num_visualized_samples, :vis_frames]
                    )
                    vis_action_tokens = action_tokens[
                        :num_visualized_samples, :vis_transition_count
                    ]
                    image_path = f"{batch_dir}/next_frame_comparison_grid.png"
                    save_comparison_images_next_frame(
                        predicted_videos,
                        vis_action_tokens.squeeze(-1).detach().cpu().numpy().tolist(),
                        expected_videos,
                        batch_dir,
                    )
                    saved_reconstructed_image_paths.append(image_path)
                    saved_visualization_samples += num_visualized_samples

                if (
                    vis_frames >= 2
                    and num_visualized_samples > 0
                    and predict_action_residuals
                    and reconstructed_residuals is not None
                ):
                    target_residuals = video_batch[:, 1:, :, :, :] - video_batch[:, :-1, :, :, :]
                    target_residual_videos = convert_video_to_images(
                        target_residuals[
                            :num_visualized_samples, :vis_transition_count
                        ],
                        value_mode="signed_residual",
                    )
                    predicted_residual_videos = convert_video_to_images(
                        reconstructed_residuals[
                            :num_visualized_samples, :vis_transition_count
                        ],
                        value_mode="signed_residual",
                    )
                    residual_image_path = (
                        f"{batch_dir}/residual_comparison_grid.png"
                    )
                    save_residual_comparison_images(
                        predicted_residual_videos,
                        target_residual_videos,
                        vis_action_tokens.squeeze(-1).detach().cpu().numpy().tolist(),
                        expected_videos,
                        batch_dir,
                        file_suffix="residual_comparison_grid.png",
                    )
                    saved_residual_image_paths.append(residual_image_path)

                if experiment_logger and image_path is not None:
                    experiment_logger.log_image(
                        f"{split}/comparison_{batch_idx}",
                        image_path,
                        step=global_step,
                    )
                    if residual_image_path is not None:
                        experiment_logger.log_image(
                            f"{split}/residual_comparison_{batch_idx}",
                            residual_image_path,
                            step=global_step,
                        )
                if image_path is not None:
                    logger.debug(
                        "Saved comparison images to "
                        f"{image_path}"
                        + (
                            f" and {residual_image_path}"
                            if residual_image_path is not None
                            else ""
                        )
                    )

            except Exception as e:
                traceback.print_exc()
                logging.warning(f"Error in {split} evaluation batch {batch_idx}: {e}")
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
            f"{split}/fid": frechet_distance,
            f"{split}/fvd": fvd_score,
        }

    # Compute average PSNR metrics
    psnr_metrics = {}
    if psnr_inferred_list:
        avg_psnr_inferred = sum(psnr_inferred_list) / len(psnr_inferred_list)
        avg_psnr_random = sum(psnr_random_list) / len(psnr_random_list)
        avg_delta_psnr = sum(delta_psnr_list) / len(delta_psnr_list)
        psnr_metrics = {
            f"{split}/psnr_inferred": avg_psnr_inferred,
            f"{split}/psnr_random": avg_psnr_random,
            f"{split}/delta_psnr": avg_delta_psnr,
        }
        logger.info(
            f"PSNR metrics: inferred={avg_psnr_inferred:.4f}, random={avg_psnr_random:.4f}, "
            f"delta={avg_delta_psnr:.4f}"
        )

    # Pixel similarity between the last two frames (t-2 vs t-1).
    frame_sim_metrics: dict[str, float] = {}
    if gt_next_frame_sim_list:
        frame_sim_metrics[f"{split}/ground_truth_next_frame_sim"] = sum(
            gt_next_frame_sim_list
        ) / len(gt_next_frame_sim_list)
    if pred_next_frame_sim_list:
        frame_sim_metrics[f"{split}/predicted_next_frame_sim"] = sum(
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
        codebook_metrics = {f"{split}/action_codebook_{k}": v for k, v in usage.items()}
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
            f"{split}/{k}": sum(d[k] for d in residual_coverage_accum) / len(residual_coverage_accum)
            for k in keys
        }
        logger.info(
            f"Residual coverage: R²={residual_metrics[f'{split}/residual_r2']:.4f}, "
            f"cosine={residual_metrics[f'{split}/residual_cosine']:.4f}, "
            f"changed_px_mse={residual_metrics[f'{split}/changed_pixel_mse']:.6f}, "
            f"changed_frac={residual_metrics[f'{split}/changed_pixel_fraction']:.4f}"
        )

    # Build log string with available metrics
    log_parts = [f"token_loss={avg_token_loss:.6f}", f"action_loss={avg_action_loss:.6f}"]
    if frechet_metrics:
        log_parts.append(f"FID={frechet_metrics.get(f'{split}/fid', float('nan')):.4f}")
        log_parts.append(f"FVD={frechet_metrics.get(f'{split}/fvd', float('nan')):.4f}")
    if psnr_metrics:
        log_parts.append(f"delta_PSNR={psnr_metrics.get(f'{split}/delta_psnr', float('nan')):.4f}")
    if frame_sim_metrics:
        log_parts.append(
            f"gt_sim={frame_sim_metrics.get(f'{split}/ground_truth_next_frame_sim', float('nan')):.4f}"
        )
        log_parts.append(
            f"pred_sim={frame_sim_metrics.get(f'{split}/predicted_next_frame_sim', float('nan')):.4f}"
        )
    if residual_metrics:
        log_parts.append(f"residual_R²={residual_metrics.get(f'{split}/residual_r2', float('nan')):.4f}")
    if codebook_metrics:
        log_parts.append(
            f"codebook_use={codebook_metrics.get(f'{split}/action_codebook_usage_fraction', float('nan')):.2%}"
        )
        log_parts.append(
            f"codebook_ppl={codebook_metrics.get(f'{split}/action_codebook_perplexity', float('nan')):.2f}"
        )
    if predict_action_residuals:
        log_parts.append(
            "saved "
            f"{len(saved_residual_image_paths)} residual and "
            f"{len(saved_reconstructed_image_paths)} reconstructed images"
        )
    else:
        log_parts.append(
            f"saved {len(saved_reconstructed_image_paths)} reconstructed images"
        )
    
    logger.info(f"{split.replace('_', ' ').capitalize()} complete: {', '.join(log_parts)}")

    # Log to wandb if available
    if experiment_logger and global_step is not None:
        log_dict = {
            f"{split}/token_loss": avg_token_loss,
            f"{split}/action_loss": avg_action_loss,
            f"{split}/total_loss": avg_token_loss + avg_action_loss,
            f"{split}/epoch": epoch,
            **frechet_metrics,
            **psnr_metrics,
            **frame_sim_metrics,
            **residual_metrics,
            **codebook_metrics,
        }
        experiment_logger.log(log_dict, step=global_step)

        # Log comparison images in batches of 5, stacked vertically
        if saved_reconstructed_image_paths:
            experiment_logger.log_image_batches(
                key_prefix=f"{split}/comparison",
                image_paths=saved_reconstructed_image_paths,
                batch_size=5,
                step=global_step,
            )
        if predict_action_residuals and saved_residual_image_paths:
            experiment_logger.log_image_batches(
                key_prefix=f"{split}/residual_comparison",
                image_paths=saved_residual_image_paths,
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
    experiment_logger: ExperimentLogger | None = None,
    save_dir: str = "dynamics_model_results",
    split: str = "eval",
) -> None:
    """Run teacher-forced and autoregressive rollout evals.

    Expects ``rollout_dataloader`` to yield clips of length
    ``2 * config.num_images_in_video`` (2T). For each clip the function:

    1. Extracts GT actions for the full 2T clip via the action model.
    2. Saves a teacher-forced comparison grid from ``model.inference`` over the
       entire 2T clip, with predictions beginning at frame ``T - 1``.
    3. Saves an autoregressive rollout grid from ``model.rollout`` seeded by
       a *single* GT frame and predicting the remaining ``2T - 1`` frames.
       This stresses the model's ability to extrapolate beyond its trained
       context length of ``T`` frames; the comparison grid renders the full
       2T-frame horizon so context degradation past frame ``T`` is visible.

    ``split`` controls the prefix used for log keys and the on-disk directory
    (``"eval"`` for the test dataloader, ``"train"`` for the train dataloader).
    """
    if split not in ("eval", "train"):
        raise ValueError(f"split must be 'eval' or 'train', got {split!r}")

    model.eval()
    T = config.num_images_in_video
    saved_rollout_image_paths: list[str] = []
    saved_teacher_forced_image_paths: list[str] = []
    saved_visualization_samples = 0

    # Collect predicted / real / prev-frame slices across batches so we can
    # compute residual-coverage, PSNR, FID and FVD on the full prediction
    # horizon at the end. ``prev`` for each predicted frame is the GT frame
    # that immediately precedes it; this matches the "copy previous frame"
    # baseline used by ``compute_residual_coverage``.
    rollout_pred_batches: list[torch.Tensor] = []
    rollout_real_batches: list[torch.Tensor] = []
    rollout_prev_batches: list[torch.Tensor] = []
    teacher_forced_pred_batches: list[torch.Tensor] = []
    teacher_forced_real_batches: list[torch.Tensor] = []
    teacher_forced_prev_batches: list[torch.Tensor] = []

    rollout_key = f"{split}_rollout"
    teacher_forced_key = f"{split}_teacher_forced"

    eval_dir = f"{save_dir}/{rollout_key}/epoch_{epoch}/step_{global_step}"
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(
        f"Running {split} rollout eval ({config.rollout_eval_batches} batches) → {eval_dir}"
    )

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(rollout_dataloader):
            if batch_idx >= config.rollout_eval_batches:
                break

            try:
                video_batch = video_batch.to(device, non_blocking=True)  # (B, 2T, C, H, W)

                # Extract GT actions for the full 2T clip
                action_encoded = model.action_model.encode(video_batch)
                actions_full = model.action_model.get_action_sequence(
                    action_encoded
                )  # (B, 2T-1)

                teacher_forced_full = model.inference(
                    video_batch,
                    actions_full[:, -1],
                    max_steps=config.rollout_max_denoising_steps,
                )  # (B, 2T, C, H, W)

                # Seed with a *single* GT frame and roll out across the full
                # 2T-frame horizon to stress how predictions degrade as the
                # autoregressive context grows past the trained window of T
                # frames. ``actions_full`` already contains all 2T-1
                # transitions (index 0 is the transition into frame 1), which
                # is exactly what ``rollout`` needs.
                predicted_full = model.rollout(
                    video_batch[:, :1],
                    actions_full,
                    max_steps=config.rollout_max_denoising_steps,
                )  # (B, 2T, C, H, W)

                # Autoregressive rollout predictions are at frames 1..2T-1.
                # The previous GT frame for step k (k=0..2T-2) is the one at
                # index k, i.e. video_batch[:, :2T-1].
                rollout_pred_batches.append(
                    predicted_full[:, 1:].detach().cpu()
                )
                rollout_real_batches.append(
                    video_batch[:, 1:].detach().cpu()
                )
                rollout_prev_batches.append(
                    video_batch[:, : 2 * T - 1].detach().cpu()
                )

                # Teacher-forced predictions are at frames T-1..2T-1 (T+1
                # frames, one per sliding T-frame window). The previous GT
                # frame for predicted frame at index k is at index k-1, so
                # prev spans T-2..2T-2 inclusive == video_batch[:, T-2:2T-1].
                teacher_forced_pred_batches.append(
                    teacher_forced_full[:, T - 1 :].detach().cpu()
                )
                teacher_forced_real_batches.append(
                    video_batch[:, T - 1 :].detach().cpu()
                )
                teacher_forced_prev_batches.append(
                    video_batch[:, T - 2 : 2 * T - 1].detach().cpu()
                )

                visualization_slots = _remaining_visualization_slots(
                    saved_visualization_samples
                )
                num_visualized_samples = min(video_batch.shape[0], visualization_slots)
                if num_visualized_samples > 0:
                    batch_dir = f"{eval_dir}/batch_{batch_idx}"
                    os.makedirs(batch_dir, exist_ok=True)

                    teacher_window, teacher_prediction_start = (
                        _prediction_boundary_window(
                            video_batch.shape[1],
                            T - 1,
                        )
                    )
                    teacher_forced_actions = _slice_actions_for_rollout_grid(
                        actions_full[:num_visualized_samples, T - 2 :],
                        T - 1,
                        teacher_window,
                    )
                    gt_teacher_images = convert_video_to_images(
                        video_batch[:num_visualized_samples, teacher_window]
                    )
                    teacher_forced_images = convert_video_to_images(
                        teacher_forced_full[:num_visualized_samples, teacher_window]
                    )
                    save_rollout_comparison_grid(
                        gt_videos=gt_teacher_images,
                        predicted_videos=teacher_forced_images,
                        predicted_actions=teacher_forced_actions.detach()
                        .cpu()
                        .numpy()
                        .tolist(),
                        output_dir=batch_dir,
                        prediction_start_idx=teacher_prediction_start,
                        file_suffix="teacher_forced_comparison_grid.png",
                    )

                    # Render the full 2T-frame rollout so context extension
                    # past the trained window of T frames is visible. The
                    # rollout is seeded from frame 0 alone, so predictions
                    # begin at frame 1.
                    rollout_window = slice(0, video_batch.shape[1])
                    rollout_prediction_start = 1
                    rollout_actions = _slice_actions_for_rollout_grid(
                        actions_full[:num_visualized_samples],
                        rollout_prediction_start,
                        rollout_window,
                    )
                    gt_rollout_images = convert_video_to_images(
                        video_batch[:num_visualized_samples, rollout_window]
                    )
                    rollout_images = convert_video_to_images(
                        predicted_full[:num_visualized_samples, rollout_window]
                    )
                    save_rollout_comparison_grid(
                        gt_videos=gt_rollout_images,
                        predicted_videos=rollout_images,
                        predicted_actions=rollout_actions.detach()
                        .cpu()
                        .numpy()
                        .tolist(),
                        output_dir=batch_dir,
                        prediction_start_idx=rollout_prediction_start,
                    )

                    rollout_image_path = f"{batch_dir}/rollout_comparison_grid.png"
                    teacher_forced_image_path = (
                        f"{batch_dir}/teacher_forced_comparison_grid.png"
                    )
                    saved_rollout_image_paths.append(rollout_image_path)
                    saved_teacher_forced_image_paths.append(teacher_forced_image_path)
                    saved_visualization_samples += num_visualized_samples

                    if experiment_logger:
                        experiment_logger.log_image(
                            f"{rollout_key}/comparison_{batch_idx}",
                            rollout_image_path,
                            step=global_step,
                        )
                        experiment_logger.log_image(
                            f"{teacher_forced_key}/comparison_{batch_idx}",
                            teacher_forced_image_path,
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

    rollout_metrics = _aggregate_rollout_metrics(
        pred_batches=rollout_pred_batches,
        real_batches=rollout_real_batches,
        prev_batches=rollout_prev_batches,
        prefix=rollout_key,
        label="autoregressive rollout",
    )
    teacher_forced_metrics = _aggregate_rollout_metrics(
        pred_batches=teacher_forced_pred_batches,
        real_batches=teacher_forced_real_batches,
        prev_batches=teacher_forced_prev_batches,
        prefix=teacher_forced_key,
        label="teacher-forced rollout",
    )

    logger.info(
        f"{split} rollout eval complete: saved "
        f"{len(saved_rollout_image_paths)} rollout grids and "
        f"{len(saved_teacher_forced_image_paths)} teacher-forced grids"
    )

    if experiment_logger:
        if rollout_metrics or teacher_forced_metrics:
            experiment_logger.log(
                {**rollout_metrics, **teacher_forced_metrics},
                step=global_step,
            )
        if saved_rollout_image_paths:
            experiment_logger.log_image_batches(
                key_prefix=f"{rollout_key}/comparison",
                image_paths=saved_rollout_image_paths,
                batch_size=5,
                step=global_step,
            )
        if saved_teacher_forced_image_paths:
            experiment_logger.log_image_batches(
                key_prefix=f"{teacher_forced_key}/comparison",
                image_paths=saved_teacher_forced_image_paths,
                batch_size=5,
                step=global_step,
            )

    model.train()


def run_evaluation_suite(
    model: DynamicsModel,
    test_dataloader: VideoWindowLoader,
    train_rollout_dataloader: VideoWindowLoader | None,
    rollout_dataloader: VideoWindowLoader | None,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: DynamicsModelTrainingConfig,
    experiment_logger: ExperimentLogger | None,
    label: str,
    run_rollout: bool,
) -> float:
    """Run the full eval suite once and return eval total loss.

    When ``run_rollout`` is False the (expensive) autoregressive +
    teacher-forced rollout evals are skipped; only the standard eval and
    train_eval losses/metrics are computed.
    """
    eval_token_loss, eval_action_loss = evaluate_model(
        model,
        test_dataloader,
        device,
        epoch,
        global_step,
        config,
        experiment_logger=experiment_logger,
        save_dir=config.save_dir,
        split="eval",
    )
    eval_loss = eval_token_loss + eval_action_loss
    logger.info(
        f"{label} eval - Token Loss: {eval_token_loss:.6f}, "
        f"Action Loss: {eval_action_loss:.6f}, Total: {eval_loss:.6f}"
    )

    if train_rollout_dataloader is not None:
        train_eval_token_loss, train_eval_action_loss = evaluate_model(
            model,
            train_rollout_dataloader,
            device,
            epoch,
            global_step,
            config,
            experiment_logger=experiment_logger,
            save_dir=config.save_dir,
            split="train_eval",
            num_frames=config.num_images_in_video,
        )
        logger.info(
            f"{label} train eval - Token Loss: {train_eval_token_loss:.6f}, "
            f"Action Loss: {train_eval_action_loss:.6f}"
        )

    if run_rollout:
        if rollout_dataloader is not None:
            evaluate_model_rollout(
                model,
                rollout_dataloader,
                device,
                epoch,
                global_step,
                config,
                experiment_logger=experiment_logger,
                save_dir=config.save_dir,
                split="eval",
            )
        if train_rollout_dataloader is not None:
            evaluate_model_rollout(
                model,
                train_rollout_dataloader,
                device,
                epoch,
                global_step,
                config,
                experiment_logger=experiment_logger,
                save_dir=config.save_dir,
                split="train",
            )
    else:
        logger.info(
            f"Skipping rollout eval at step {global_step} "
            f"(rollout runs every {config.eval_interval * config.rollout_every_n_evals} steps)"
        )

    return eval_loss


def train_epoch(
    dynamics_model: DynamicsModel,
    action_model: LatentActionVQVAE,
    train_dataloader: VideoWindowLoader,
    test_dataloader: VideoWindowLoader,
    rollout_dataloader: VideoWindowLoader | None,
    train_rollout_dataloader: VideoWindowLoader | None,
    dynamics_optimizer: optim.Optimizer,
    dynamics_scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: DynamicsModelTrainingConfig,
    experiment_logger: ExperimentLogger | None,
    global_step: int,
    total_steps: int,
    best_loss: float,
    start_batch: int = 0,
    save_dir: str = "dynamics_model_results",
):
    """Train for one epoch"""
    total_loss = 0.0
    total_optimizer_step_loss = 0.0
    num_batches = len(train_dataloader)
    accumulation_steps = config.gradient_accumulation_steps
    T = config.num_images_in_video

    # Configure the resumable dataloader: set the per-epoch shuffle and skip
    # ahead to ``start_batch`` at the sampler level, so workers never load
    # (and discard) the skipped batches.
    train_dataloader.resumable_loader.set_epoch(epoch)
    train_dataloader.resumable_loader.set_start_batch(start_batch)

    os.makedirs(f"{save_dir}/train/epoch_{epoch}", exist_ok=True)

    epoch_start_time = time.time()
    total_optimizer_step_action_loss = 0.0
    total_optimizer_step_token_loss = 0.0
    token_loss_sum_gpu = torch.tensor(0.0, device=device)
    action_loss_sum_gpu = torch.tensor(0.0, device=device)
    microbatch_count = 0
    logged_action_tokens: list[torch.Tensor] = []
    rollout_calls_acc: list[int] = []
    eps_acc: list[float] = []
    dynamics_optimizer.zero_grad()
    max_context = dynamics_model.num_images_in_video - 1
    use_amp = config.use_bf16 and device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    for batch_idx, video_batch in enumerate(train_dataloader, start=start_batch):
        batch_start_time = time.time()

        video_batch = video_batch.to(device, non_blocking=True)
        video_batch = video_batch[:, :T + 1]

        with amp_ctx:
            action_encoded = action_model.encode(video_batch)
            action_tokens = action_model.get_action_sequence(action_encoded)

            # --- Scheduled sampling: build the prediction basis ---
            eps = inverse_sigmoid_decay(global_step / total_steps, decay_rate=10.0)
            eps_acc.append(eps)

            if config.scheduled_sampling == "bengio_per_frame":
                use_gt = torch.rand(T) < eps
                duplicate = video_batch.clone()
                rollout_calls = 0

                with torch.no_grad():
                    for t in range(T):
                        if bool(use_gt[t]):
                            continue
                        ctx = duplicate[:, : t + 1]
                        ctx_start = max(0, t - max_context + 1)
                        ctx_actions = action_tokens[:, ctx_start:t]
                        pred = dynamics_model.predict_next_frame(
                            ctx,
                            action_tokens[:, t],
                            max_steps=config.rollout_max_denoising_steps,
                            context_actions=ctx_actions if ctx_actions.shape[1] > 0 else None,
                        )
                        duplicate[:, t + 1] = pred
                        rollout_calls += 1

                video_prediction_basis = duplicate[:, 1:]
                rollout_calls_acc.append(rollout_calls)
            elif config.scheduled_sampling == "free_run_mix":
                B = video_batch.shape[0]
                with torch.no_grad():
                    prediction_basis = dynamics_model.rollout(
                        video_batch[:, :1],
                        action_tokens,
                        max_steps=config.rollout_max_denoising_steps,
                    )
                keep_gt_mask = torch.rand(B, T, 1, 1, 1, device=device) < eps
                video_prediction_basis = torch.where(
                    keep_gt_mask,
                    video_batch[:, 1:],
                    prediction_basis[:, 1:],
                )
                rollout_calls_acc.append(T)
            else:
                video_prediction_basis = video_batch[:, 1:]
                rollout_calls_acc.append(0)

            gaussian_noise = torch.randn_like(video_prediction_basis) * 0.02
            video_prediction_basis = video_prediction_basis + gaussian_noise
            clamp_video_prediction_basis = torch.clamp(video_prediction_basis, 0, 1.0)

            decoded, token_loss, action_tokens = dynamics_model(
                video_batch[:, 1:],
                video_prediction_basis,
                action_tokens[:, 1:],
            )

            reconstructed_action_video = action_model.decode(video_batch, action_encoded)
            action_loss = dynamics_model.action_loss_fn(video_batch, reconstructed_action_video)

        remaining_batches = num_batches - batch_idx
        current_window_size = min(accumulation_steps, remaining_batches)
        combined_loss = (token_loss + action_loss) / current_window_size
        combined_loss.backward()

        token_loss_sum_gpu += token_loss.detach()
        action_loss_sum_gpu += action_loss.detach()
        microbatch_count += 1
        logged_action_tokens.append(action_tokens.detach())

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
            dynamics_scheduler.step()
            dynamics_optimizer.zero_grad()

            avg_token_loss = (token_loss_sum_gpu / microbatch_count).item()
            avg_action_loss = (action_loss_sum_gpu / microbatch_count).item()
            total_loss = avg_token_loss + avg_action_loss
            total_optimizer_step_loss += total_loss
            total_optimizer_step_action_loss += avg_action_loss
            total_optimizer_step_token_loss += avg_token_loss
            action_token_window = (
                torch.cat([tokens.reshape(-1) for tokens in logged_action_tokens], dim=0).cpu()
                if logged_action_tokens
                else torch.empty(0, dtype=torch.long)
            )
            action_codebook_usage = compute_codebook_usage(
                action_token_window, action_model.action_vocab_size
            )
            top_code_counts = get_top_code_counts(
                action_token_window, action_model.action_vocab_size
            )

            batch_time = time.time() - batch_start_time
            global_step += 1

            avg_rollout_calls = sum(rollout_calls_acc) / len(rollout_calls_acc)
            avg_eps = sum(eps_acc) / len(eps_acc)

            if experiment_logger:
                current_lrs = dynamics_scheduler.get_last_lr()
                training_metrics = {
                    "train/loss": total_loss,
                    "train/token_loss": avg_token_loss,
                    "train/action_loss": avg_action_loss,
                    "train/learning_rate": current_lrs[0],
                    "train/dynamics_learning_rate": current_lrs[0],
                    "train/action_learning_rate": current_lrs[1],
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "train/action_codebook_usage_fraction": action_codebook_usage["usage_fraction"],
                    "train/action_codebook_perplexity": action_codebook_usage["perplexity"],
                    "train/action_codebook_normalized_entropy": action_codebook_usage[
                        "normalized_entropy"
                    ],
                    "train/action_codebook_num_unique": action_codebook_usage["num_unique"],
                    "train/action_codebook_num_tokens": action_codebook_usage["num_tokens"],
                    "train/scheduled_sampling_eps": avg_eps,
                    "train/rollout_calls": avg_rollout_calls,
                }
                experiment_logger.log(training_metrics, step=global_step)

            if global_step % config.log_interval == 0:
                current_lrs = dynamics_scheduler.get_last_lr()
                logger.info(
                    f"Step {global_step}, Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {total_loss:.6f} (token: {avg_token_loss:.6f}, action: {avg_action_loss:.6f}), "
                    f"Codebook: unique={int(action_codebook_usage['num_unique'])}/{action_model.action_vocab_size} "
                    f"({action_codebook_usage['usage_fraction']:.2%}), "
                    f"ppl={action_codebook_usage['perplexity']:.2f}, "
                    f"top_counts=[{format_top_code_counts(top_code_counts)}], "
                    f"LRs: dynamics={current_lrs[0]:.2e}, action={current_lrs[1]:.2e}, "
                    f"eps={avg_eps:.3f}, "
                    f"rollout_calls={avg_rollout_calls:.2f}/{T}, "
                    f"Time: {batch_time:.2f}s"
                )

            token_loss_sum_gpu.zero_()
            action_loss_sum_gpu.zero_()
            microbatch_count = 0
            logged_action_tokens = []
            rollout_calls_acc = []
            eps_acc = []

            eval_improved = False
            if global_step % config.eval_interval == 0:
                rollout_interval_steps = (
                    config.eval_interval * config.rollout_every_n_evals
                )
                run_rollout = global_step % rollout_interval_steps == 0
                eval_loss = run_evaluation_suite(
                    dynamics_model,
                    test_dataloader,
                    train_rollout_dataloader,
                    rollout_dataloader,
                    device,
                    epoch,
                    global_step,
                    config,
                    experiment_logger,
                    label=f"Step {global_step}",
                    run_rollout=run_rollout,
                )
                eval_improved = eval_loss < best_loss
                if eval_improved:
                    best_loss = eval_loss
                    logger.info(f"New best eval loss: {best_loss:.6f}")

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
                    if experiment_logger:
                        experiment_logger.log(
                            {f"train/{k}": v for k, v in train_coverage.items()},
                            step=global_step,
                        )

                # Log action-decoder reconstruction on the current train batch,
                # matching the eval-side next-frame and residual comparison grids.
                dynamics_model.eval()
                with torch.no_grad():
                    train_action_encoded = dynamics_model.action_model.encode(video_batch)
                    train_action_decoded = dynamics_model.action_model.decode(
                        video_batch, train_action_encoded
                    )
                    train_action_tokens = dynamics_model.action_model.get_action_sequence(
                        train_action_encoded
                    )
                    if dynamics_model.predict_action_residuals:
                        train_residuals = train_action_decoded
                        train_next_frames = reconstruct_predicted_frames_from_residuals(
                            video_batch, train_residuals
                        )
                    else:
                        train_residuals = None
                        train_next_frames = train_action_decoded

                    train_batch_dir = f"{save_dir}/train/epoch_{epoch}/batch_{batch_idx}"
                    os.makedirs(train_batch_dir, exist_ok=True)
                    train_vis_frames = min(
                        VISUALIZATION_SEQUENCE_FRAMES,
                        video_batch.shape[1],
                    )
                    train_image_path = (
                        f"{train_batch_dir}/action_decoder_next_frame_comparison_grid.png"
                    )

                    train_residual_image_path: str | None = None
                    train_num_visualized_samples = min(
                        video_batch.shape[0],
                        MAX_EVAL_SAVED_IMAGES,
                    )
                    if train_vis_frames >= 2 and train_num_visualized_samples > 0:
                        train_vis_transition_count = train_vis_frames - 1
                        train_expected_videos = convert_video_to_images(
                            video_batch[
                                :train_num_visualized_samples, :train_vis_frames
                            ]
                        )
                        train_predicted_videos = convert_video_to_images(
                            train_next_frames[
                                :train_num_visualized_samples,
                                :train_vis_transition_count,
                            ]
                        )
                        train_vis_action_tokens = train_action_tokens[
                            :train_num_visualized_samples,
                            :train_vis_transition_count,
                        ]
                        save_comparison_images_next_frame(
                            train_predicted_videos,
                            train_vis_action_tokens.squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist(),
                            train_expected_videos,
                            train_batch_dir,
                            file_suffix="action_decoder_next_frame_comparison_grid.png",
                        )

                    if (
                        train_vis_frames >= 2
                        and train_num_visualized_samples > 0
                        and train_residuals is not None
                    ):
                        train_target_residuals = (
                            video_batch[:, 1:, :, :, :] - video_batch[:, :-1, :, :, :]
                        )
                        train_target_residual_videos = convert_video_to_images(
                            train_target_residuals[
                                :train_num_visualized_samples,
                                :train_vis_transition_count,
                            ],
                            value_mode="signed_residual",
                        )
                        train_residual_videos = convert_video_to_images(
                            train_residuals[
                                :train_num_visualized_samples,
                                :train_vis_transition_count,
                            ],
                            value_mode="signed_residual",
                        )
                        train_residual_image_path = (
                            f"{train_batch_dir}/action_decoder_residual_comparison_grid.png"
                        )
                        save_residual_comparison_images(
                            train_residual_videos,
                            train_target_residual_videos,
                            train_vis_action_tokens.squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist(),
                            train_expected_videos,
                            train_batch_dir,
                            file_suffix="action_decoder_residual_comparison_grid.png",
                        )
                dynamics_model.train()

                logger.info(
                    "Saved train action-decoder reconstruction grid to "
                    f"{train_image_path}"
                    + (
                        f" and {train_residual_image_path}"
                        if train_residual_image_path is not None
                        else ""
                    )
                )
                if experiment_logger:
                    experiment_logger.log_image(
                        "train/action_decoder_comparison",
                        train_image_path,
                        step=global_step,
                    )
                    if train_residual_image_path is not None:
                        experiment_logger.log_image(
                            "train/action_decoder_residual_comparison",
                            train_residual_image_path,
                            step=global_step,
                        )

            if global_step % config.save_interval == 0:
                save_checkpoint(
                    dynamics_model,
                    dynamics_optimizer,
                    dynamics_scheduler,
                    epoch,
                    batch_idx,
                    total_loss,
                    config,
                    best_loss,
                    train_dataloader.get_state(),
                    action_model=action_model,
                    is_best=eval_improved,
                    global_step=global_step,
                )

    # Calculate average loss over optimizer steps
    num_batches_processed = max(num_batches - start_batch, 0)
    num_optimizer_steps = math.ceil(num_batches_processed / accumulation_steps)
    avg_loss = total_optimizer_step_loss / max(num_optimizer_steps, 1)
    avg_action_loss = total_optimizer_step_action_loss / max(num_optimizer_steps, 1)
    avg_token_loss = total_optimizer_step_token_loss / max(num_optimizer_steps, 1)
    epoch_time = time.time() - epoch_start_time

    # Log epoch summary
    if experiment_logger:
        epoch_metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_token_loss": avg_token_loss,
            "train/epoch_action_loss": avg_action_loss,
            "train/epoch_time": epoch_time,
            "train/epoch": epoch,
        }
        experiment_logger.log(epoch_metrics, step=global_step)

    logger.info(
        f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f} "
        f"(token: {avg_token_loss:.6f}, action: {avg_action_loss:.6f}), "
        f"Time: {epoch_time:.2f}s"
    )
    return avg_loss, global_step, best_loss


def main(config: DynamicsModelTrainingConfig):
    """Main training function"""

    # Validate required config
    if not config.tokenizer_checkpoint_path:
        raise ValueError(
            "tokenizer_checkpoint_path is required to load pretrained tokenizer"
        )
    if config.log_interval <= 0:
        raise ValueError("log_interval must be greater than 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if config.save_interval <= 0:
        raise ValueError("save_interval must be greater than 0")
    if config.rollout_every_n_evals <= 0:
        raise ValueError("rollout_every_n_evals must be greater than 0")

    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"maskgit_{timestamp}"

    logging_backend = resolve_logging_backend(
        logging_backend=config.logging_backend,
        use_wandb=config.use_wandb,
    )
    experiment_logger = ExperimentLogger(
        backend=logging_backend,
        run_name=config.experiment_name,
        config_summary=config.__dict__,
        group="mask-git-training",
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_tags=config.wandb_tags,
        wandb_notes=config.wandb_notes,
        tensorboard_dir=config.tensorboard_dir,
    )

    logger.info(f"Starting MaskGIT training - Experiment: {config.experiment_name}")
    logger.info(f"Using S3: {config.use_s3}")
    logger.info(f"Logging backend: {logging_backend}")
    if experiment_logger and experiment_logger.tensorboard_log_dir is not None:
        logger.info(f"TensorBoard log dir: {experiment_logger.tensorboard_log_dir}")

    # Enable TF32 and optimized matmul on supported GPUs (H100, A100, etc.)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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

    if config.use_compile and device.type == "cuda":
        _compile_mode = "default"
        model.decoder = torch.compile(model.decoder, mode=_compile_mode, dynamic=True)  # type: ignore[assignment]
        action_model.encoder = torch.compile(action_model.encoder, mode=_compile_mode, dynamic=True)  # type: ignore[assignment]
        action_model.decoder_transformer = torch.compile(action_model.decoder_transformer, mode=_compile_mode, dynamic=True)  # type: ignore[assignment]
        logger.info("torch.compile enabled on decoder and action model submodules")

    # Create data loader
    logger.info("Creating data loader...")
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")

    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )

    # Build the train dataset with 2T-frame windows so a single underlying
    # dataset can be reused for both training (trimmed to T frames per batch)
    # and the train-side rollout eval (full 2T frames). This avoids creating a
    # second small "train rollout" subset that the model could overfit to.
    rollout_window = 2 * config.num_images_in_video
    logger.info(
        f"Building train dataset with 2T window={rollout_window} "
        "(trimmed to T at training time)..."
    )
    train_dataset, rollout_dataset = build_datasets(
        config,
        local_cache,
        num_frames_in_video=rollout_window,
        train_limit=None,
        test_limit=100,
    )
    _, test_dataset = build_datasets(config, local_cache)

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

    rollout_dataloader: VideoWindowLoader = VideoWindowLoader(
        dataset=rollout_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=4,
        seed=config.seed,
    )
    logger.info(f"Rollout eval dataset (test): {len(rollout_dataset)} samples")

    # Train rollout eval: a second dataloader over the SAME train_dataset so we
    # don't build a separate small fixed "train rollout" subset.
    train_rollout_dataloader: VideoWindowLoader = VideoWindowLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=4,
        seed=config.seed,
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

    # Watch model with wandb
    if experiment_logger:
        experiment_logger.watch(model, log="all", log_freq=config.log_interval * 10)

    # Create optimizer and scheduler
    action_params = [
        parameter for parameter in action_model.parameters() if parameter.requires_grad
    ]
    action_param_ids = {id(parameter) for parameter in action_params}
    tokenizer_params = [
        parameter
        for parameter in model.tokenizer.parameters()
    ]
    tokenizer_param_ids = {id(parameter) for parameter in tokenizer_params}

    # Only optimize the dynamics model parameters that are not shared with the action model or tokenizer
    dynamics_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in action_param_ids and id(parameter) not in tokenizer_param_ids
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
        f"action_lr={config.action_learning_rate}\n"
        f"Num params dynamics model: {sum(p.numel() for p in dynamics_params)}\n"
        f"Num params action model: {sum(p.numel() for p in action_params)}\n"
        f"Num params tokenizer: {sum(p.numel() for p in tokenizer_params)}"
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
    global_step = 0

    if config.dynamics_model_checkpoint_path:
        model, optimizer, scheduler, checkpoint = load_checkpoint(
            config.dynamics_model_checkpoint_path, model, optimizer, scheduler, device
        )
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint.get("batch_idx", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        checkpoint_global_step = checkpoint.get("global_step")
        global_step = (
            checkpoint_global_step
            if checkpoint_global_step is not None
            else max(0, scheduler.last_epoch)
        )

        # End-of-epoch checkpoints store the last seen batch index, so advance to
        # the next epoch instead of re-entering a completed one.
        if start_batch >= len(train_dataloader):
            start_epoch += 1
            start_batch = 0

        logger.info(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    logger.info(
        f"Evaluating every {config.eval_interval} optimizer steps "
        f"(rollout eval every {config.eval_interval * config.rollout_every_n_evals} steps); "
        f"saving every {config.save_interval} optimizer steps."
    )

    # Training loop
    logger.info("Starting training loop...")

    try:
        model.train()
        last_completed_epoch = start_epoch - 1
        last_avg_loss = float("inf")
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss, global_step, best_loss = train_epoch(
                model,
                action_model,
                train_dataloader,
                test_dataloader,
                rollout_dataloader,
                train_rollout_dataloader,
                optimizer,
                scheduler,
                device,
                epoch,
                config,
                experiment_logger,
                global_step,
                total_steps,
                best_loss,
                epoch_start_batch,
                config.save_dir,
            )
            last_completed_epoch = epoch
            last_avg_loss = avg_loss

        if last_completed_epoch >= start_epoch:
            final_eval_is_best = False
            final_eval_ran = False
            if global_step % config.eval_interval == 0:
                logger.info(
                    f"Final eval already ran at scheduled step {global_step}; "
                    "skipping duplicate final eval."
                )
            else:
                final_eval_loss = run_evaluation_suite(
                    model,
                    test_dataloader,
                    train_rollout_dataloader,
                    rollout_dataloader,
                    device,
                    last_completed_epoch,
                    global_step,
                    config,
                    experiment_logger,
                    label=f"Final step {global_step}",
                    run_rollout=True,
                )
                final_eval_ran = True
                final_eval_is_best = final_eval_loss < best_loss
                if final_eval_is_best:
                    best_loss = final_eval_loss
                    logger.info(f"New best eval loss: {best_loss:.6f}")

            if not final_eval_ran and global_step % config.save_interval == 0:
                logger.info(
                    f"Final checkpoint already saved at scheduled step {global_step}; "
                    "skipping duplicate final checkpoint."
                )
            else:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    last_completed_epoch,
                    len(train_dataloader),
                    last_avg_loss,
                    config,
                    best_loss,
                    train_dataloader.get_state(),
                    action_model=action_model,
                    is_best=final_eval_is_best,
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
            avg_loss,
            config,
            best_loss,
            dataloader_state,
            action_model=action_model,
            global_step=global_step,
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        if experiment_logger:
            experiment_logger.finish()

    logger.info("Training completed!")
    logger.info(f"Best loss achieved: {best_loss:.6f}")

    # if config.use_s3 and s3_manager:
        # logger.info(f"Checkpoints saved to S3 bucket: {s3_manager.bucket_name}")
    # else:
        # logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")

    if logging_backend == "wandb":
        logger.info(f"Training metrics logged to Wandb project: {config.wandb_project}")
    elif logging_backend == "tensorboard" and experiment_logger is not None:
        logger.info(
            "Training metrics logged to TensorBoard dir: "
            f"{experiment_logger.tensorboard_log_dir}"
        )


if __name__ == "__main__":
    config = tyro.cli(DynamicsModelTrainingConfig)
    logger.info(f"Starting training... config: {config.__dict__}")
    main(config)
    logger.info("Training completed!")
