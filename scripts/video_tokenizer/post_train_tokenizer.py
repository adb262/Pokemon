"""Post-train tokenizer encoder/decoder on dynamics-model rollout codes.

Loads pretrained tokenizer, action model, and dynamics model checkpoints.
Freezes the action/dynamics stack. Autoregressively rolls out future frame
codes through the frozen dynamics model, decodes them with the *trainable*
tokenizer decoder, and supervises against ground-truth frames.

Joint loss:
  reconstruction_loss: sliding window -> encoder -> FSQ -> decoder -> MSE vs GT
  rollout_loss:        encoder context -> frozen dynamics -> codes -> decoder -> MSE vs GT

Usage::

    python -m scripts.video_tokenizer.post_train_tokenizer \
        --tokenizer_checkpoint_path <path> \
        --dynamics_model_checkpoint_path <path> \
        --action_model_checkpoint_path <path> \
        --post_train_frames 32 \
        --dynamics_context_frames 16 \
        --batch_size 4 \
        --num_epochs 5
"""
import csv
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tyro
from matplotlib.ticker import LogFormatterSciNotation
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from dynamics_model.checkpoints import adapt_state_dict_to_model
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.model import DynamicsModel
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from loss.loss_fns import reconstruction_loss
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.experiment_logger import ExperimentLogger, resolve_logging_backend
from monitoring.residual_coverage import compute_residual_coverage
from monitoring.videos import convert_video_to_images, save_rollout_comparison_grid
from video_tokenization.checkpoints import (
    load_model_from_checkpoint,
    save_checkpoint,
)
from video_tokenization.model import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PostTrainTokenizerConfig:
    tokenizer_checkpoint_path: str = ""
    dynamics_tokenizer_checkpoint_path: Optional[str] = None
    dynamics_model_checkpoint_path: str = ""
    action_model_checkpoint_path: Optional[str] = None

    # Dataset
    dataset_type: Literal["pokemon", "atari_pong"] = "atari_pong"
    atari_pong_data_dir: Optional[str] = "data/atari_pong"
    atari_pong_crop_scoreboard: bool = False
    atari_pong_require_full_gameplay: bool = True
    frames_dir: str = "pokemon_frames"
    dataset_train_key: Optional[str] = None
    sync_from_s3: bool = False
    use_s3: bool = False
    local_cache_dir: Optional[str] = os.environ.get("BT_RW_CACHE_DIR", "cache")
    max_cache_size: int = 100000
    frame_spacing: int = 1
    num_unique_frames: Optional[int] = None
    dataset_limit: int = 500000
    test_dataset_limit: Optional[int] = 1000

    # Architecture (read from tokenizer checkpoint, but available for override)
    image_size: int = 84
    patch_size: int = 4
    num_images_in_video: int = 5
    bins: list[int] = field(default_factory=lambda: [8, 8, 6, 5])

    # Post-training specifics
    post_train_frames: int = 32
    dynamics_context_frames: int = 16
    rollout_seed_frames: int = 1
    max_denoising_steps: int = 1
    reconstruction_loss_weight: float = 1.0
    rollout_loss_weight: float = 1.0
    train_encoder: bool = True

    # Dynamics model architecture (read from checkpoint config)
    action_bins: list[int] = field(default_factory=lambda: [6, 4])
    action_d_model: int = 256
    action_num_transformer_layers: int = 4
    action_num_heads: int = 2
    action_latent_dim: int = 64
    dynamics_d_model: int = 512
    dynamics_num_transformer_layers: int = 8
    dynamics_num_heads: int = 8
    predict_action_residuals: bool = False
    action_decoder_loss: Literal["l2", "clipped_l2"] = "l2"
    action_l2_clip_c: float = 10.0
    dynamics_token_loss: Literal["ce", "clipped_ce"] = "ce"
    dynamics_ce_clip_c: float = 0.03

    # Optimizer
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-7
    num_epochs: int = 5
    warmup_steps: int = 100
    seed: int = 42

    # Logging / checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_dir: str = "post_train_tokenizer_checkpoints"
    save_dir: str = "post_train_tokenizer_results"
    tensorboard_dir: str = "tokenizer_runs"
    experiment_name: Optional[str] = None
    logging_backend: Literal["wandb", "tensorboard", "none"] = "tensorboard"
    use_wandb: bool = False
    wandb_project: str = "pokemon-post-train-tokenizer"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    wandb_notes: Optional[str] = None
    max_comparison_images: int = 5
    reconstruction_error_scale: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16: bool = True
    use_compile: bool = True
    eval_only: bool = False
    compare_tokenizer_checkpoint_path: Optional[str] = None
    tokenizer_label: str = "post-trained"
    compare_tokenizer_label: str = "Base Tokenizer"
    comparison_eval_batches: Optional[int] = None

    # Early stopping
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint_config(checkpoint_path: Optional[str]) -> dict:
    if checkpoint_path is None:
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config", {})


def _apply_dynamics_checkpoint_config(
    config: PostTrainTokenizerConfig,
    dynamics_ckpt_config: dict,
    action_ckpt_config: dict,
) -> None:
    """Override architecture fields from checkpoint configs in-place."""
    field_map = {
        "action_bins": "action_bins",
        "action_d_model": "action_d_model",
        "action_num_transformer_layers": "action_num_transformer_layers",
        "action_num_heads": "action_num_heads",
        "action_latent_dim": "action_latent_dim",
        "dynamics_d_model": "dynamics_d_model",
        "dynamics_num_transformer_layers": "dynamics_num_transformer_layers",
        "dynamics_num_heads": "dynamics_num_heads",
        "predict_action_residuals": "predict_action_residuals",
        "action_decoder_loss": "action_decoder_loss",
        "action_l2_clip_c": "action_l2_clip_c",
        "dynamics_token_loss": "dynamics_token_loss",
        "dynamics_ce_clip_c": "dynamics_ce_clip_c",
    }
    for key, attr in field_map.items():
        if key in dynamics_ckpt_config:
            setattr(config, attr, dynamics_ckpt_config[key])
    for key in ("action_bins", "action_d_model", "action_num_transformer_layers",
                "action_num_heads", "action_latent_dim"):
        if key in action_ckpt_config:
            setattr(config, key, action_ckpt_config[key])
    for key in ("num_images_in_video", "image_size", "patch_size", "frame_spacing",
                "num_unique_frames", "dataset_type", "atari_pong_data_dir",
                "atari_pong_crop_scoreboard", "atari_pong_require_full_gameplay"):
        if key in dynamics_ckpt_config:
            setattr(config, key, dynamics_ckpt_config[key])


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def _safe_psnr_from_mse(mse: float) -> float:
    if mse <= 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@dataclass
class EarlyStoppingState:
    best_loss: float = float("inf")
    evals_without_improvement: int = 0

    def update(self, eval_loss: float, min_delta: float) -> bool:
        if eval_loss < self.best_loss - min_delta:
            self.best_loss = eval_loss
            self.evals_without_improvement = 0
            return True
        self.evals_without_improvement += 1
        return False

    def should_stop(self, patience: int) -> bool:
        return patience > 0 and self.evals_without_improvement >= patience


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_dynamics_config(config: PostTrainTokenizerConfig) -> DynamicsModelTrainingConfig:
    dynamics_tokenizer_checkpoint_path = (
        config.dynamics_tokenizer_checkpoint_path
        if config.dynamics_tokenizer_checkpoint_path is not None
        else config.tokenizer_checkpoint_path
    )
    return DynamicsModelTrainingConfig(
        tokenizer_checkpoint_path=dynamics_tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=config.dynamics_model_checkpoint_path,
        action_model_checkpoint_path=config.action_model_checkpoint_path,
        num_images_in_video=config.dynamics_context_frames,
        image_size=config.image_size,
        patch_size=config.patch_size,
        action_bins=list(config.action_bins),
        action_d_model=config.action_d_model,
        action_num_transformer_layers=config.action_num_transformer_layers,
        action_num_heads=config.action_num_heads,
        action_latent_dim=config.action_latent_dim,
        dynamics_d_model=config.dynamics_d_model,
        dynamics_num_transformer_layers=config.dynamics_num_transformer_layers,
        dynamics_num_heads=config.dynamics_num_heads,
        predict_action_residuals=config.predict_action_residuals,
        action_decoder_loss=config.action_decoder_loss,
        action_l2_clip_c=config.action_l2_clip_c,
        dynamics_token_loss=config.dynamics_token_loss,
        dynamics_ce_clip_c=config.dynamics_ce_clip_c,
        device=config.device,
        use_wandb=False,
        batch_size=config.batch_size,
        dataset_type=config.dataset_type,
        atari_pong_data_dir=config.atari_pong_data_dir,
        atari_pong_crop_scoreboard=config.atari_pong_crop_scoreboard,
        atari_pong_require_full_gameplay=config.atari_pong_require_full_gameplay,
        frames_dir=config.frames_dir,
        local_cache_dir=config.local_cache_dir,
        use_s3=config.use_s3,
        dataset_train_key=config.dataset_train_key,
        sync_from_s3=config.sync_from_s3,
        frame_spacing=config.frame_spacing,
        num_unique_frames=config.num_unique_frames,
        dataset_limit=config.dataset_limit,
        rollout_max_denoising_steps=config.max_denoising_steps,
    )


def load_frozen_dynamics_stack(
    config: PostTrainTokenizerConfig,
    device: torch.device,
) -> DynamicsModel:
    """Load the full dynamics stack with a *separate* frozen tokenizer copy."""
    dynamics_cfg = _build_dynamics_config(config)
    tokenizer_checkpoint_path = (
        config.dynamics_tokenizer_checkpoint_path
        if config.dynamics_tokenizer_checkpoint_path is not None
        else config.tokenizer_checkpoint_path
    )

    frozen_tokenizer, _ = load_model_from_checkpoint(
        tokenizer_checkpoint_path, device
    )
    _freeze_module(frozen_tokenizer)

    action_model = create_action_model_from_dynamics_config(dynamics_cfg).to(device)
    action_model.eval()

    dynamics_model = create_dynamics_model(
        dynamics_cfg, frozen_tokenizer, action_model
    ).to(device)

    if config.dynamics_model_checkpoint_path:
        checkpoint = torch.load(
            config.dynamics_model_checkpoint_path, map_location=device
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_to_model(state_dict, dynamics_model)
        missing, unexpected = dynamics_model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            logger.warning(
                "Missing %d keys loading dynamics checkpoint (first 5: %s)",
                len(missing), missing[:5],
            )
        if unexpected:
            logger.warning(
                "Unexpected %d keys loading dynamics checkpoint (first 5: %s)",
                len(unexpected), unexpected[:5],
            )
        logger.info(
            "Loaded dynamics model weights from %s",
            config.dynamics_model_checkpoint_path,
        )

    _freeze_module(dynamics_model)
    return dynamics_model


# ---------------------------------------------------------------------------
# Autoregressive rollout producing codes + trainable decode
# ---------------------------------------------------------------------------

def _build_rolling_tokenizer_history(
    generated_pixels: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    """Return the generated-frame history for a tokenizer rolling window."""
    if generated_pixels.shape[1] < 1:
        raise ValueError("Tokenizer history requires at least one generated frame")
    if context_frames < 2:
        raise ValueError(
            f"Tokenizer decoder window must contain at least 2 frames, got {context_frames}"
        )

    previous_context_frames = context_frames - 1
    history = generated_pixels[:, -previous_context_frames:]
    pad_frames = previous_context_frames - history.shape[1]

    if pad_frames > 0:
        seed_padding = generated_pixels[:, :1].expand(-1, pad_frames, -1, -1, -1)
        history = torch.cat([seed_padding, history], dim=1)

    return history


def _build_rolling_gt_action_window(
    video_batch: torch.Tensor,
    frame_idx: int,
    window_frames: int,
) -> torch.Tensor:
    """Return a fixed-size GT window ending at ``frame_idx`` for action encoding."""
    if frame_idx < 1:
        raise ValueError(f"frame_idx must be at least 1, got {frame_idx}")
    if window_frames < 2:
        raise ValueError(
            f"Action window must contain at least 2 frames, got {window_frames}"
        )

    end_idx = frame_idx + 1
    start_idx = max(0, end_idx - window_frames)
    action_window = video_batch[:, start_idx:end_idx]
    pad_frames = window_frames - action_window.shape[1]

    if pad_frames > 0:
        seed_padding = video_batch[:, :1].expand(-1, pad_frames, -1, -1, -1)
        action_window = torch.cat([seed_padding, action_window], dim=1)

    return action_window


@torch.no_grad()
def _get_rolling_gt_action_tokens(
    dynamics_model: DynamicsModel,
    video_batch: torch.Tensor,
    target_frame_idx: int,
    context_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Encode GT actions from the same fixed-size rolling windows used in training.

    Returns the context actions for transitions inside the generated dynamics
    context, plus the next action for the target frame ``frame_idx``.
    """
    action_window_frames = dynamics_model.action_model.num_images_in_video
    available_actions = action_window_frames - 1
    if context_size < 1:
        raise ValueError(f"context_size must be at least 1, got {context_size}")
    if context_size > available_actions:
        raise ValueError(
            "Current dynamics context needs more actions than the action model "
            f"window can provide: context_size={context_size}, "
            f"available_actions={available_actions}"
        )

    action_window = _build_rolling_gt_action_window(
        video_batch,
        target_frame_idx,
        action_window_frames,
    )
    action_encoded = dynamics_model.action_model.encode(action_window)
    action_tokens = dynamics_model.action_model.get_action_sequence(
        action_encoded
    ).long()
    rolling_tokens = action_tokens[:, -context_size:]
    context_actions = rolling_tokens[:, :-1] if context_size > 1 else None
    next_action = rolling_tokens[:, -1]
    return context_actions, next_action


@torch.no_grad()
def _dynamics_predict_next_frame_codes(
    dynamics_model: DynamicsModel,
    context_pixels: torch.Tensor,
    action: torch.Tensor,
    max_steps: int,
    context_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the frozen dynamics MaskGIT sampler and return predicted codes.

    Returns shape ``(B, num_patches)`` of integer code indices for the
    predicted next frame.
    """
    max_context = dynamics_model.num_images_in_video - 1
    context_window = context_pixels[:, -max_context:]
    placeholder = context_window[:, -1:].clone()
    inference_window = torch.cat([context_window, placeholder], dim=1)

    n_frames = inference_window.shape[1]
    targets = dynamics_model.tokenizer.quantized_value_to_codes(
        dynamics_model.tokenizer.encode(inference_window)
    ).long()

    batch_size, _, num_patches = targets.shape

    action_token = action.long().to(targets.device)
    if action_token.dim() == 1:
        action_token = action_token.unsqueeze(1)

    targets[:, -1, :] = dynamics_model.tokenizer.get_mask_token_idx()

    expected_context = n_frames - 2
    if context_actions is None:
        if expected_context > 0:
            action_video_encoded = dynamics_model.action_model.encode(
                inference_window[:, :-1]
            )
            context_action_tokens = dynamics_model.action_model.get_action_sequence(
                action_video_encoded
            ).long()
        else:
            context_action_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=targets.device
            )
    else:
        context_action_tokens = context_actions.long().to(targets.device)
        if context_action_tokens.shape != (batch_size, expected_context):
            raise ValueError(
                f"context_actions must have shape (B, {expected_context}), "
                f"got {tuple(context_action_tokens.shape)}"
            )

    action_tokens = torch.cat([context_action_tokens, action_token], dim=1)
    action_embeddings = dynamics_model.action_embedding(action_tokens.long())

    step = 0
    mask_locations = torch.full(
        (batch_size, num_patches), True, dtype=torch.bool, device=targets.device
    )
    while step < max_steps:
        x = dynamics_model.tokenizer_embedding(targets.long())
        x[:, :-1, :] += action_embeddings.unsqueeze(2)
        x = dynamics_model.decoder(x)
        logits = dynamics_model.vocab_head(x)

        if not mask_locations.any():
            break

        ratio_remaining = dynamics_model.cosine_scheduler(
            max_steps, step + 1, targets.device
        )
        target_still_masked = int(num_patches * ratio_remaining)
        current_masked = int(mask_locations.sum(dim=1).amax().item())
        tokens_to_update = max(0, current_masked - target_still_masked)

        probs = dynamics_model.softmax(logits[:, -1])
        mask_flat = mask_locations.view(-1)
        probs_flat = probs.view(-1, probs.size(-1))
        samples_flat = torch.full(
            (batch_size * num_patches,), -100,
            device=targets.device, dtype=torch.long,
        )
        if mask_flat.any():
            probs_masked = probs_flat[mask_flat]
            probs_masked = probs_masked / (probs_masked.sum(-1, keepdim=True) + 1e-9)
            sampled_masked = torch.multinomial(
                probs_masked, 1, replacement=False
            ).squeeze(-1)
            samples_flat[mask_flat] = sampled_masked

        samples = samples_flat.view(batch_size, num_patches)
        sampled_scores = probs.gather(-1, samples.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        sampled_scores[samples < 0] = -1

        if tokens_to_update > 0:
            _, top_positions = torch.topk(sampled_scores, tokens_to_update, dim=-1)
            top_tokens = samples.gather(1, top_positions)
            targets_last = targets[:, -1, :]
            targets_last = targets_last.scatter(1, top_positions, top_tokens)
            targets[:, -1, :] = targets_last
            mask_locations = mask_locations.scatter(
                1, top_positions,
                torch.zeros_like(top_positions, dtype=torch.bool),
            )

        step += 1

    return targets[:, -1, :]


def rollout_with_trainable_decoder(
    dynamics_model: DynamicsModel,
    trainable_tokenizer: VideoTokenizer,
    video_batch: torch.Tensor,
    config: PostTrainTokenizerConfig,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a closed-loop autoregressive rollout with the trainable decoder.

    The dynamics model predicts the next frame's codes from generated pixels.
    The tokenizer decoder then renders those codes in a rolling tokenizer window
    whose context codes are encoded from previously generated frames only.

    Returns:
        rollout_decoded: (B, rollout_steps, C, H, W) decoded predicted frames
        recon_decoded:   (B, total_frames, C, H, W) sliding-window reconstruction
        gt_future:       (B, rollout_steps, C, H, W) ground-truth future frames
    """
    seed_frames = config.rollout_seed_frames
    total_frames = min(config.post_train_frames, video_batch.shape[1])
    dynamics_ctx = config.dynamics_context_frames
    tokenizer_ctx = trainable_tokenizer.decoder.num_images_in_video
    amp_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
    )

    generated_pixels = video_batch[:, :seed_frames].clone()

    rollout_decoded_frames: list[torch.Tensor] = []
    recon_decoded_frames: list[torch.Tensor] = []

    for frame_idx in range(total_frames):
        if frame_idx < seed_frames:
            ctx_start = max(0, frame_idx - tokenizer_ctx + 1)
            recon_window = video_batch[:, ctx_start : frame_idx + 1]
            with amp_ctx:
                quantized = trainable_tokenizer.encode(recon_window)
                decoded_window = trainable_tokenizer.decode(quantized)
            recon_decoded_frames.append(decoded_window[:, -1])
            continue

        current_num = generated_pixels.shape[1]
        ctx_size = min(current_num, dynamics_ctx - 1)
        ctx_start_gen = current_num - ctx_size
        context_pixels = generated_pixels[:, ctx_start_gen:]

        ctx_actions, next_action = _get_rolling_gt_action_tokens(
            dynamics_model,
            video_batch,
            frame_idx,
            ctx_size,
        )

        with torch.no_grad():
            next_codes = _dynamics_predict_next_frame_codes(
                dynamics_model,
                context_pixels,
                next_action,
                config.max_denoising_steps,
                context_actions=ctx_actions,
            )

        ctx_start_recon = max(0, frame_idx - tokenizer_ctx + 1)
        recon_window_gt = video_batch[:, ctx_start_recon : frame_idx + 1]
        with amp_ctx:
            quantized_recon = trainable_tokenizer.encode(recon_window_gt)
            decoded_recon_window = trainable_tokenizer.decode(quantized_recon)
        recon_decoded_frames.append(decoded_recon_window[:, -1])

        tokenizer_history = _build_rolling_tokenizer_history(
            generated_pixels,
            tokenizer_ctx,
        )
        with amp_ctx:
            quantized_context = trainable_tokenizer.encode(tokenizer_history)
        context_codes = trainable_tokenizer.quantized_value_to_codes(
            quantized_context.detach()
        )
        all_codes = torch.cat(
            [context_codes, next_codes.unsqueeze(1)], dim=1
        )
        with amp_ctx:
            decoded_rollout_window = trainable_tokenizer.decode_from_codes(all_codes)
        next_frame_decoded = decoded_rollout_window[:, -1]

        rollout_decoded_frames.append(next_frame_decoded)
        generated_pixels = torch.cat(
            [generated_pixels, next_frame_decoded.detach().unsqueeze(1)], dim=1
        )

    rollout_decoded = torch.stack(rollout_decoded_frames, dim=1)
    recon_decoded = torch.stack(recon_decoded_frames, dim=1)
    gt_future = video_batch[:, seed_frames:total_frames]

    return rollout_decoded, recon_decoded, gt_future


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _make_frame_triptych(
    expected: Image.Image,
    predicted: Image.Image,
    error: Image.Image,
) -> Image.Image:
    images = [expected.convert("RGB"), predicted.convert("RGB"), error.convert("RGB")]
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    triptych = Image.new("RGB", (width, height), color=(0, 0, 0))
    x_offset = 0
    for img in images:
        triptych.paste(img, (x_offset, 0))
        x_offset += img.width
    return triptych


def _save_rollout_triptych_grid(
    predicted_videos: list[list[Image.Image]],
    expected_videos: list[list[Image.Image]],
    error_videos: list[list[Image.Image]],
    output_path: str,
    seed_frames: int,
    dynamics_context_frames: int | None = None,
    title: str = "Each cell: GT | Rollout Pred | Abs Err",
) -> str:
    if not predicted_videos:
        raise ValueError("Cannot save rollout grid for an empty batch")

    seq_idx = 0
    num_frames = len(predicted_videos[seq_idx])
    cols = min(8, max(1, num_frames))
    rows = math.ceil(num_frames / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.0))
    flat_axes = np.asarray(axs).reshape(-1)
    for frame_idx, ax in enumerate(flat_axes):
        ax.axis("off")
        if frame_idx >= num_frames:
            continue
        triptych = _make_frame_triptych(
            expected_videos[seq_idx][frame_idx],
            predicted_videos[seq_idx][frame_idx],
            error_videos[seq_idx][frame_idx],
        )
        ax.imshow(triptych, interpolation="nearest")
        label = f"Frame {frame_idx}"
        if frame_idx < seed_frames:
            label += " (seed)"
        elif (
            dynamics_context_frames is not None
            and frame_idx == dynamics_context_frames
        ):
            label += " (context limit)"
        ax.set_title(label)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_per_frame_metrics_plot(
    mse_by_frame: list[float],
    output_path: str,
    first_frame_idx: int,
    dynamics_context_frames: int,
    tokenizer_context_frames: int,
    action_window_frames: int,
) -> str:
    frames = list(range(first_frame_idx, first_frame_idx + len(mse_by_frame)))
    min_positive_mse = 1e-12
    mse_for_plot = [max(m, min_positive_mse) for m in mse_by_frame]
    psnr_values = [_safe_psnr_from_mse(m) for m in mse_by_frame]
    rollout_steps = list(range(1, len(mse_by_frame) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(rollout_steps, mse_for_plot, marker="o", label="Sliding rollout")
    axs[0].set_title("MSE as rollout window moves")
    axs[0].set_xlabel("Rollout step (target frame advances by 1)")
    axs[0].set_ylabel("MSE")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    axs[0].grid(True, alpha=0.3)
    axs[0].grid(True, which="minor", alpha=0.15)
    axs[0].legend()

    axs[1].plot(rollout_steps, psnr_values, marker="o", label="Sliding rollout")
    axs[1].set_title("PSNR as rollout window moves")
    axs[1].set_xlabel("Rollout step (target frame advances by 1)")
    axs[1].set_ylabel("PSNR (dB)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    first_full_dynamics_step = max(1, dynamics_context_frames - first_frame_idx)
    first_full_tokenizer_step = max(1, tokenizer_context_frames - first_frame_idx)
    first_full_action_step = max(1, action_window_frames - first_frame_idx)
    for ax in axs:
        ax.secondary_xaxis(
            "top",
            functions=(
                lambda step: step + first_frame_idx - 1,
                lambda frame: frame - first_frame_idx + 1,
            ),
        ).set_xlabel("Target frame index")
        for step, label in (
            (first_full_dynamics_step, "dynamics window full"),
            (first_full_tokenizer_step, "tokenizer window full"),
            (first_full_action_step, "action window full"),
        ):
            if 1 <= step <= len(mse_by_frame):
                ax.axvline(step, linestyle="--", linewidth=1, alpha=0.35)
                ax.text(
                    step,
                    0.98,
                    label,
                    rotation=90,
                    va="top",
                    ha="right",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    alpha=0.7,
                )

    fig.suptitle(
        "Sliding-window rollout metrics: generated context and GT action window "
        "advance one frame per point",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_per_frame_metrics_csv(
    mse_by_frame: list[float],
    output_path: str,
    first_frame_idx: int,
    dynamics_context_frames: int,
    tokenizer_context_frames: int,
    action_window_frames: int,
) -> str:
    rows = []
    for offset, mse in enumerate(mse_by_frame):
        target_frame_idx = first_frame_idx + offset
        rows.append(
            {
                "rollout_step": offset + 1,
                "target_frame_idx": target_frame_idx,
                "dynamics_context_start": max(0, target_frame_idx - dynamics_context_frames + 1),
                "dynamics_context_end": target_frame_idx - 1,
                "tokenizer_context_start": max(0, target_frame_idx - tokenizer_context_frames + 1),
                "tokenizer_context_end": target_frame_idx - 1,
                "action_window_start": max(0, target_frame_idx - action_window_frames + 1),
                "action_window_end": target_frame_idx,
                "mse": mse,
                "psnr": _safe_psnr_from_mse(mse),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _save_comparison_metrics_plot(
    metrics_by_label: dict[str, list[float]],
    output_path: str,
    first_frame_idx: int,
    dynamics_context_frames: int,
    action_window_frames: int,
) -> str:
    rollout_steps = None
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for label, mse_by_frame in metrics_by_label.items():
        if rollout_steps is None:
            rollout_steps = list(range(1, len(mse_by_frame) + 1))
        mse_for_plot = [max(m, 1e-12) for m in mse_by_frame]
        psnr_values = [_safe_psnr_from_mse(m) for m in mse_by_frame]
        axs[0].plot(rollout_steps, mse_for_plot, marker="o", label=label)
        axs[1].plot(rollout_steps, psnr_values, marker="o", label=label)

    axs[0].set_title("MSE by rollout step")
    axs[0].set_xlabel("Rollout step (target frame advances by 1)")
    axs[0].set_ylabel("MSE")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    axs[1].set_title("PSNR by rollout step")
    axs[1].set_xlabel("Rollout step (target frame advances by 1)")
    axs[1].set_ylabel("PSNR (dB)")

    first_full_dynamics_step = max(1, dynamics_context_frames - first_frame_idx)
    first_full_action_step = max(1, action_window_frames - first_frame_idx)
    full_window_labels_by_step: dict[int, list[str]] = {}
    for step, label in (
        (first_full_dynamics_step, "dynamics"),
        (first_full_action_step, "action"),
    ):
        full_window_labels_by_step.setdefault(step, []).append(label)

    for ax in axs:
        ax.secondary_xaxis(
            "top",
            functions=(
                lambda step: step + first_frame_idx - 1,
                lambda frame: frame - first_frame_idx + 1,
            ),
        ).set_xlabel("Target frame index")
        for step, labels in full_window_labels_by_step.items():
            if rollout_steps is not None and 1 <= step <= len(rollout_steps):
                ax.axvline(step, linestyle="--", linewidth=1, alpha=0.35)
                label = " + ".join(labels)
                label += " window full" if len(labels) == 1 else " windows full"
                ax.text(
                    step,
                    0.98,
                    label,
                    rotation=90,
                    va="top",
                    ha="right",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    alpha=0.7,
                )
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Pre/post tokenizer comparison with rolling dynamics context", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_comparison_metrics_csv(
    metrics_by_label: dict[str, list[float]],
    output_path: str,
    first_frame_idx: int,
) -> str:
    rows = []
    for label, mse_by_frame in metrics_by_label.items():
        for offset, mse in enumerate(mse_by_frame):
            rows.append(
                {
                    "label": label,
                    "rollout_step": offset + 1,
                    "target_frame_idx": first_frame_idx + offset,
                    "mse": mse,
                    "psnr": _safe_psnr_from_mse(mse),
                }
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def post_train_epoch(
    trainable_tokenizer: VideoTokenizer,
    dynamics_model: DynamicsModel,
    train_dataloader: VideoWindowLoader,
    test_dataloader: VideoWindowLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: PostTrainTokenizerConfig,
    experiment_logger: ExperimentLogger,
    early_stopping_state: EarlyStoppingState,
    start_batch: int = 0,
) -> tuple[float, int, bool]:
    accumulation_steps = config.gradient_accumulation_steps
    num_batches = len(train_dataloader)
    total_loss = 0.0
    should_early_stop = False

    train_dataloader.resumable_loader.set_epoch(epoch)
    train_dataloader.resumable_loader.set_start_batch(start_batch)

    use_amp = config.use_bf16 and device.type == "cuda"
    amp_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
    )

    accumulated_loss_gpu = torch.tensor(0.0, device=device)
    microbatch_count = 0
    global_step = epoch * (num_batches // accumulation_steps)

    for batch_idx, video_batch in enumerate(train_dataloader, start=start_batch):
        video_batch = video_batch.to(device, non_blocking=True)
        total_frames = min(config.post_train_frames, video_batch.shape[1])
        video_batch = video_batch[:, :total_frames]

        rollout_decoded, recon_decoded, gt_future = rollout_with_trainable_decoder(
            dynamics_model, trainable_tokenizer, video_batch, config, use_amp,
        )

        with amp_ctx:
            recon_loss = reconstruction_loss(video_batch, recon_decoded)

            rollout_loss = reconstruction_loss(gt_future, rollout_decoded)

            loss = (
                config.reconstruction_loss_weight * recon_loss
                + config.rollout_loss_weight * rollout_loss
            ) / accumulation_steps

        loss.backward()
        accumulated_loss_gpu += loss.detach() * accumulation_steps
        microbatch_count += 1

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(
                trainable_tokenizer.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = (accumulated_loss_gpu / microbatch_count).item()
            total_loss += avg_loss
            global_step = epoch * (num_batches // accumulation_steps) + (
                batch_idx // accumulation_steps
            )

            if experiment_logger:
                experiment_logger.log(
                    {
                        "post_train/loss": avg_loss,
                        "post_train/reconstruction_loss": recon_loss.detach().item(),
                        "post_train/rollout_loss": rollout_loss.detach().item(),
                        "post_train/learning_rate": scheduler.get_last_lr()[0],
                        "post_train/epoch": epoch,
                        "post_train/batch": batch_idx,
                    },
                    step=global_step,
                )

            if (batch_idx // accumulation_steps) % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {avg_loss:.6f} "
                    f"(recon: {recon_loss.item():.6f}, rollout: {rollout_loss.item():.6f}), "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            accumulated_loss_gpu.zero_()
            microbatch_count = 0

            if global_step > 0 and global_step % config.eval_interval == 0:
                eval_loss = eval_post_trained_decoder(
                    trainable_tokenizer, dynamics_model, test_dataloader,
                    device, epoch, config, experiment_logger,
                    global_step=global_step,
                )
                improved = early_stopping_state.update(
                    eval_loss, config.early_stopping_min_delta
                )
                if improved:
                    logger.info(
                        f"New best eval loss: {early_stopping_state.best_loss:.6f} "
                        f"(step {global_step})"
                    )

                save_checkpoint(
                    trainable_tokenizer, optimizer, scheduler,
                    epoch, batch_idx, avg_loss, _as_tokenizer_config(config),
                    early_stopping_state.best_loss,
                    train_dataloader.get_state(),
                )

                if early_stopping_state.should_stop(config.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered at step {global_step}"
                    )
                    should_early_stop = True
                    break

                trainable_tokenizer.train()
                if not config.train_encoder:
                    trainable_tokenizer.encoder.eval()

    num_opt_steps = max(num_batches // accumulation_steps, 1)
    avg_epoch_loss = total_loss / num_opt_steps
    return avg_epoch_loss, global_step, should_early_stop


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout_mse_by_frame(
    trainable_tokenizer: VideoTokenizer,
    dynamics_model: DynamicsModel,
    test_dataloader: VideoWindowLoader,
    device: torch.device,
    config: PostTrainTokenizerConfig,
    max_batches: Optional[int] = None,
) -> list[float]:
    trainable_tokenizer.eval()
    use_amp = config.use_bf16 and device.type == "cuda"
    rollout_frame_sse = None
    rollout_frame_count = None

    for batch_idx, video_batch in enumerate(test_dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        video_batch = video_batch.to(device, non_blocking=True)
        total_frames = min(config.post_train_frames, video_batch.shape[1])
        video_batch = video_batch[:, :total_frames]

        rollout_decoded, _, gt_future = rollout_with_trainable_decoder(
            dynamics_model, trainable_tokenizer, video_batch, config, use_amp,
        )

        diff = (rollout_decoded.float().clamp(0, 1) - gt_future.float().clamp(0, 1)) ** 2
        frame_sse = diff.sum(dim=(0, 2, 3, 4)).detach().cpu()
        frame_count = torch.full(
            (diff.shape[1],),
            diff.shape[0] * diff.shape[2] * diff.shape[3] * diff.shape[4],
            dtype=torch.float64,
        )
        if rollout_frame_sse is None or rollout_frame_count is None:
            rollout_frame_sse = frame_sse.double()
            rollout_frame_count = frame_count
        else:
            rollout_frame_sse += frame_sse.double()
            rollout_frame_count += frame_count

    if rollout_frame_sse is None or rollout_frame_count is None:
        raise ValueError("No batches were available for rollout comparison")

    trainable_tokenizer.train()
    if not config.train_encoder:
        trainable_tokenizer.encoder.eval()

    return (
        rollout_frame_sse / torch.clamp(rollout_frame_count, min=1)
    ).tolist()


@torch.no_grad()
def compare_tokenizer_rollout_metrics(
    primary_tokenizer: VideoTokenizer,
    comparison_tokenizer: VideoTokenizer,
    dynamics_model: DynamicsModel,
    test_dataloader: VideoWindowLoader,
    device: torch.device,
    config: PostTrainTokenizerConfig,
) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.save_dir}/tokenizer_comparison/{timestamp}"
    tokenizers_by_label = {
        config.tokenizer_label: primary_tokenizer,
        config.compare_tokenizer_label: comparison_tokenizer,
    }
    accumulators: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        label: (None, None) for label in tokenizers_by_label
    }
    use_amp = config.use_bf16 and device.type == "cuda"

    for batch_idx, video_batch in enumerate(test_dataloader):
        if (
            config.comparison_eval_batches is not None
            and batch_idx >= config.comparison_eval_batches
        ):
            break

        video_batch = video_batch.to(device, non_blocking=True)
        total_frames = min(config.post_train_frames, video_batch.shape[1])
        video_batch = video_batch[:, :total_frames]

        for label, tokenizer in tokenizers_by_label.items():
            tokenizer.eval()
            rollout_decoded, _, gt_future = rollout_with_trainable_decoder(
                dynamics_model, tokenizer, video_batch, config, use_amp,
            )
            diff = (
                rollout_decoded.float().clamp(0, 1)
                - gt_future.float().clamp(0, 1)
            ) ** 2
            frame_sse = diff.sum(dim=(0, 2, 3, 4)).detach().cpu()
            frame_count = torch.full(
                (diff.shape[1],),
                diff.shape[0] * diff.shape[2] * diff.shape[3] * diff.shape[4],
                dtype=torch.float64,
            )
            rollout_frame_sse, rollout_frame_count = accumulators[label]
            if rollout_frame_sse is None or rollout_frame_count is None:
                accumulators[label] = (frame_sse.double(), frame_count)
            else:
                accumulators[label] = (
                    rollout_frame_sse + frame_sse.double(),
                    rollout_frame_count + frame_count,
                )

    metrics_by_label: dict[str, list[float]] = {}
    for label, (rollout_frame_sse, rollout_frame_count) in accumulators.items():
        if rollout_frame_sse is None or rollout_frame_count is None:
            raise ValueError("No batches were available for rollout comparison")
        metrics_by_label[label] = (
            rollout_frame_sse / torch.clamp(rollout_frame_count, min=1)
        ).tolist()

    plot_path = f"{output_dir}/pre_post_mse_psnr_comparison.png"
    csv_path = f"{output_dir}/pre_post_mse_psnr_comparison.csv"
    _save_comparison_metrics_plot(
        metrics_by_label,
        plot_path,
        config.rollout_seed_frames,
        config.dynamics_context_frames,
        dynamics_model.action_model.num_images_in_video,
    )
    _save_comparison_metrics_csv(
        metrics_by_label,
        csv_path,
        config.rollout_seed_frames,
    )
    logger.info("Saved tokenizer comparison plot to %s", plot_path)
    logger.info("Saved tokenizer comparison CSV to %s", csv_path)
    return plot_path, csv_path


@torch.no_grad()
def eval_post_trained_decoder(
    trainable_tokenizer: VideoTokenizer,
    dynamics_model: DynamicsModel,
    test_dataloader: VideoWindowLoader,
    device: torch.device,
    epoch: int,
    config: PostTrainTokenizerConfig,
    experiment_logger: ExperimentLogger,
    global_step: int = 0,
) -> float:
    trainable_tokenizer.eval()
    total_rollout_loss = 0.0
    total_recon_loss = 0.0
    total_samples = 0
    saved_image_count = 0
    use_amp = config.use_bf16 and device.type == "cuda"

    rollout_frame_sse = None
    rollout_frame_count = None
    residual_metrics_accum: list[dict[str, float]] = []
    saved_comparison_paths: list[str] = []

    eval_dir = f"{config.save_dir}/post_train_eval/epoch_{epoch}/step_{global_step}"
    os.makedirs(eval_dir, exist_ok=True)

    for batch_idx, video_batch in enumerate(test_dataloader):
        video_batch = video_batch.to(device, non_blocking=True)
        total_frames = min(config.post_train_frames, video_batch.shape[1])
        video_batch = video_batch[:, :total_frames]

        rollout_decoded, recon_decoded, gt_future = rollout_with_trainable_decoder(
            dynamics_model, trainable_tokenizer, video_batch, config, use_amp,
        )

        recon_loss = reconstruction_loss(video_batch, recon_decoded)
        rollout_loss = reconstruction_loss(gt_future, rollout_decoded)

        total_recon_loss += recon_loss.item() * video_batch.size(0)
        total_rollout_loss += rollout_loss.item() * video_batch.size(0)
        total_samples += video_batch.size(0)

        # Per-frame MSE accumulation
        diff = (rollout_decoded.float().clamp(0, 1) - gt_future.float().clamp(0, 1)) ** 2
        frame_sse = diff.sum(dim=(0, 2, 3, 4)).detach().cpu()
        frame_count = torch.full(
            (diff.shape[1],),
            diff.shape[0] * diff.shape[2] * diff.shape[3] * diff.shape[4],
            dtype=torch.float64,
        )
        if rollout_frame_sse is None or rollout_frame_count is None:
            rollout_frame_sse = frame_sse.double()
            rollout_frame_count = frame_count
        else:
            rollout_frame_sse += frame_sse.double()
            rollout_frame_count += frame_count

        if gt_future.shape[1] >= 2:
            residual_metrics_accum.append(
                compute_residual_coverage(
                    gt_frame=gt_future[:, -1],
                    pred_frame=rollout_decoded[:, -1],
                    prev_frame=gt_future[:, -2],
                )
            )

        if saved_image_count < config.max_comparison_images:
            n = min(1, video_batch.shape[0])
            full_rollout = torch.cat(
                [video_batch[:n, : config.rollout_seed_frames], rollout_decoded[:n]],
                dim=1,
            )
            gt_full = video_batch[:n, :total_frames]
            with torch.no_grad():
                vis_action_encoded = dynamics_model.action_model.encode(gt_full)
                vis_action_tokens = dynamics_model.action_model.get_action_sequence(
                    vis_action_encoded
                ).long()
                vis_rollout_actions = vis_action_tokens[
                    :, config.rollout_seed_frames - 1 : total_frames - 1
                ]
                frozen_dynamics_rollout = dynamics_model.rollout(
                    gt_full[:, : config.rollout_seed_frames],
                    vis_rollout_actions,
                    max_steps=config.max_denoising_steps,
                )
            vis_action_lists = vis_rollout_actions.detach().cpu().tolist()

            # Triptych grid (GT | Pred | Abs Error per frame)
            pred_images = convert_video_to_images(full_rollout)
            gt_images = convert_video_to_images(gt_full)
            error_images = convert_video_to_images(
                full_rollout - gt_full,
                value_mode="magnitude",
                residual_scale=config.reconstruction_error_scale,
            )
            triptych_path = f"{eval_dir}/rollout_triptych_batch_{batch_idx}.png"
            _save_rollout_triptych_grid(
                pred_images, gt_images, error_images, triptych_path,
                config.rollout_seed_frames,
                config.dynamics_context_frames,
                "Trainable tokenizer decode: GT | Pred | Abs Err",
            )

            # Side-by-side rollout comparison grid (GT row vs Predicted row)
            batch_dir = f"{eval_dir}/batch_{batch_idx}"
            save_rollout_comparison_grid(
                gt_videos=gt_images,
                predicted_videos=pred_images,
                predicted_actions=vis_action_lists,
                output_dir=batch_dir,
                prediction_start_idx=config.rollout_seed_frames,
                file_suffix="rollout_comparison_grid.png",
            )
            comparison_path = f"{batch_dir}/rollout_comparison_grid.png"
            saved_comparison_paths.append(comparison_path)

            frozen_pred_images = convert_video_to_images(frozen_dynamics_rollout)
            frozen_error_images = convert_video_to_images(
                frozen_dynamics_rollout - gt_full,
                value_mode="magnitude",
                residual_scale=config.reconstruction_error_scale,
            )
            frozen_triptych_path = (
                f"{eval_dir}/frozen_dynamics_rollout_triptych_batch_{batch_idx}.png"
            )
            _save_rollout_triptych_grid(
                frozen_pred_images,
                gt_images,
                frozen_error_images,
                frozen_triptych_path,
                config.rollout_seed_frames,
                config.dynamics_context_frames,
                "Frozen dynamics rollout: GT | Pred | Abs Err",
            )
            save_rollout_comparison_grid(
                gt_videos=gt_images,
                predicted_videos=frozen_pred_images,
                predicted_actions=vis_action_lists,
                output_dir=batch_dir,
                prediction_start_idx=config.rollout_seed_frames,
                file_suffix="frozen_dynamics_rollout_comparison_grid.png",
            )

            # Overwrite a live.png that can be refreshed in the IDE
            live_dir = f"{config.save_dir}/post_train_eval"
            os.makedirs(live_dir, exist_ok=True)
            live_comparison_path = f"{live_dir}/rollout_comparison_live.png"
            save_rollout_comparison_grid(
                gt_videos=gt_images,
                predicted_videos=pred_images,
                predicted_actions=vis_action_lists,
                output_dir=live_dir,
                prediction_start_idx=config.rollout_seed_frames,
                file_suffix="rollout_comparison_live.png",
            )
            live_triptych_path = f"{live_dir}/rollout_triptych_live.png"
            _save_rollout_triptych_grid(
                pred_images, gt_images, error_images, live_triptych_path,
                config.rollout_seed_frames,
                config.dynamics_context_frames,
                "Trainable tokenizer decode: GT | Pred | Abs Err",
            )
            frozen_live_triptych_path = (
                f"{live_dir}/frozen_dynamics_rollout_triptych_live.png"
            )
            _save_rollout_triptych_grid(
                frozen_pred_images,
                gt_images,
                frozen_error_images,
                frozen_live_triptych_path,
                config.rollout_seed_frames,
                config.dynamics_context_frames,
                "Frozen dynamics rollout: GT | Pred | Abs Err",
            )

            if experiment_logger:
                experiment_logger.log_image(
                    f"post_train_eval/rollout_comparison_{batch_idx}",
                    comparison_path,
                    step=global_step,
                )
                experiment_logger.log_image(
                    f"post_train_eval/rollout_triptych_{batch_idx}",
                    triptych_path,
                    step=global_step,
                )
                experiment_logger.log_image(
                    f"post_train_eval/frozen_dynamics_rollout_triptych_{batch_idx}",
                    frozen_triptych_path,
                    step=global_step,
                )

            saved_image_count += 1

    avg_recon_loss = total_recon_loss / max(total_samples, 1)
    avg_rollout_loss = total_rollout_loss / max(total_samples, 1)
    avg_total = avg_recon_loss + avg_rollout_loss

    if rollout_frame_sse is not None and rollout_frame_count is not None:
        mse_by_frame = (
            rollout_frame_sse / torch.clamp(rollout_frame_count, min=1)
        ).tolist()
        plot_path = f"{eval_dir}/per_frame_metrics.png"
        _save_per_frame_metrics_plot(
            mse_by_frame,
            plot_path,
            config.rollout_seed_frames,
            config.dynamics_context_frames,
            trainable_tokenizer.decoder.num_images_in_video,
            dynamics_model.action_model.num_images_in_video,
        )
        csv_path = f"{eval_dir}/per_frame_metrics.csv"
        _save_per_frame_metrics_csv(
            mse_by_frame,
            csv_path,
            config.rollout_seed_frames,
            config.dynamics_context_frames,
            trainable_tokenizer.decoder.num_images_in_video,
            dynamics_model.action_model.num_images_in_video,
        )
        logger.info("Saved per-frame metrics plot to %s", plot_path)
        logger.info("Saved per-frame metrics CSV to %s", csv_path)

    residual_summary: dict[str, float] = {}
    if residual_metrics_accum:
        keys = residual_metrics_accum[0].keys()
        residual_summary = {
            k: sum(d[k] for d in residual_metrics_accum) / len(residual_metrics_accum)
            for k in keys
        }

    logger.info(
        f"Eval step {global_step}: "
        f"recon_loss={avg_recon_loss:.6f}, rollout_loss={avg_rollout_loss:.6f}, "
        f"total={avg_total:.6f}"
    )
    if residual_summary:
        logger.info(
            f"Residual coverage: "
            f"R²={residual_summary.get('residual_r2', float('nan')):.4f}, "
            f"cosine={residual_summary.get('residual_cosine', float('nan')):.4f}, "
            f"changed_px_frac={residual_summary.get('changed_pixel_fraction', float('nan')):.4f}"
        )

    if experiment_logger:
        log_dict: dict = {
            "post_train_eval/recon_loss": avg_recon_loss,
            "post_train_eval/rollout_loss": avg_rollout_loss,
            "post_train_eval/total_loss": avg_total,
            "post_train_eval/epoch": epoch,
        }
        log_dict.update(
            {f"post_train_eval/{k}": v for k, v in residual_summary.items()}
        )
        experiment_logger.log(log_dict, step=global_step)

        if saved_comparison_paths:
            experiment_logger.log_image_batches(
                key_prefix="post_train_eval/rollout_comparison",
                image_paths=saved_comparison_paths,
                batch_size=5,
                step=global_step,
            )

    trainable_tokenizer.train()
    if not config.train_encoder:
        trainable_tokenizer.encoder.eval()

    return avg_total


# ---------------------------------------------------------------------------
# Config bridge
# ---------------------------------------------------------------------------

def _as_tokenizer_config(config: PostTrainTokenizerConfig) -> VideoTokenizerTrainingConfig:
    """Build a VideoTokenizerTrainingConfig for checkpoint saving."""
    tok_config = VideoTokenizerTrainingConfig()
    tok_config.image_size = config.image_size
    tok_config.patch_size = config.patch_size
    tok_config.num_images_in_video = config.num_images_in_video
    tok_config.bins = list(config.bins)
    tok_config.checkpoint_dir = config.checkpoint_dir
    tok_config.save_dir = config.save_dir
    tok_config.batch_size = config.batch_size
    tok_config.learning_rate = config.learning_rate
    tok_config.num_epochs = config.num_epochs
    tok_config.device = config.device
    tok_config.use_bf16 = config.use_bf16
    tok_config.experiment_name = config.experiment_name
    tok_config.dataset_type = config.dataset_type
    tok_config.local_cache_dir = config.local_cache_dir
    tok_config.seed = config.seed
    return tok_config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: PostTrainTokenizerConfig) -> None:
    if not config.tokenizer_checkpoint_path:
        raise ValueError("tokenizer_checkpoint_path is required")
    if not config.dynamics_model_checkpoint_path:
        raise ValueError("dynamics_model_checkpoint_path is required")
    if (
        config.compare_tokenizer_checkpoint_path is not None
        and config.dynamics_tokenizer_checkpoint_path is None
    ):
        config.dynamics_tokenizer_checkpoint_path = config.compare_tokenizer_checkpoint_path
        logger.info(
            "Using comparison tokenizer checkpoint for the frozen dynamics tokenizer: %s",
            config.dynamics_tokenizer_checkpoint_path,
        )

    dynamics_ckpt_config = _load_checkpoint_config(
        config.dynamics_model_checkpoint_path
    )
    action_ckpt_config = _load_checkpoint_config(config.action_model_checkpoint_path)
    _apply_dynamics_checkpoint_config(config, dynamics_ckpt_config, action_ckpt_config)

    if config.dynamics_context_frames > config.num_images_in_video:
        logger.warning(
            "dynamics_context_frames (%d) clamped to dynamics num_images_in_video (%d)",
            config.dynamics_context_frames, config.num_images_in_video,
        )
        config.dynamics_context_frames = config.num_images_in_video
    if config.post_train_frames <= config.rollout_seed_frames:
        raise ValueError(
            f"post_train_frames ({config.post_train_frames}) must be > "
            f"rollout_seed_frames ({config.rollout_seed_frames})"
        )

    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"post_train_tokenizer_{timestamp}"

    logging_backend = resolve_logging_backend(
        logging_backend=config.logging_backend,
        use_wandb=config.use_wandb,
    )
    experiment_logger = ExperimentLogger(
        backend=logging_backend,
        run_name=config.experiment_name,
        config_summary=config.__dict__,
        group="post-train-tokenizer",
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_tags=config.wandb_tags or [],
        wandb_notes=config.wandb_notes or "",
        tensorboard_dir=config.tensorboard_dir,
    )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    logger.info("Using device: %s", device)

    # Load trainable tokenizer
    logger.info("Loading trainable tokenizer from %s", config.tokenizer_checkpoint_path)
    trainable_tokenizer, tok_config = load_model_from_checkpoint(
        config.tokenizer_checkpoint_path, device
    )
    config.image_size = tok_config.image_size
    config.patch_size = tok_config.patch_size
    config.bins = list(tok_config.bins)

    # Freeze FSQ
    _freeze_module(trainable_tokenizer.fsq)

    if not config.train_encoder:
        _freeze_module(trainable_tokenizer.encoder)
        logger.info("Encoder frozen — training decoder only")
    else:
        logger.info("Training encoder + decoder jointly")

    trainable_tokenizer.train()
    if not config.train_encoder:
        trainable_tokenizer.encoder.eval()
    trainable_tokenizer.fsq.eval()

    trainable_params = [p for p in trainable_tokenizer.parameters() if p.requires_grad]
    logger.info(
        "Trainable params: %d / %d total",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in trainable_tokenizer.parameters()),
    )

    # torch.compile the trainable tokenizer submodules
    if config.use_compile and device.type == "cuda":
        _compile_mode = "default"
        trainable_tokenizer.decoder.encoder = torch.compile(
            trainable_tokenizer.decoder.encoder, mode=_compile_mode, dynamic=True,
        )  # type: ignore[assignment]
        if config.train_encoder:
            trainable_tokenizer.encoder.encoder = torch.compile(
                trainable_tokenizer.encoder.encoder, mode=_compile_mode, dynamic=True,
            )  # type: ignore[assignment]
        logger.info("torch.compile enabled on trainable tokenizer submodules")

    # Load frozen dynamics stack
    logger.info("Loading frozen dynamics stack...")
    dynamics_model = load_frozen_dynamics_stack(config, device)
    logger.info(
        "Dynamics model context window: %d frames",
        dynamics_model.num_images_in_video,
    )

    # torch.compile the frozen dynamics decoder for faster inference
    if config.use_compile and device.type == "cuda":
        dynamics_model.decoder = torch.compile(
            dynamics_model.decoder, mode="default", dynamic=True,
        )  # type: ignore[assignment]
        logger.info("torch.compile enabled on frozen dynamics decoder")

    # Dataset
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")
    local_cache = Cache(max_size=config.max_cache_size, cache_dir=config.local_cache_dir)

    train_dataset, test_dataset = build_datasets(
        config, local_cache,
        num_frames_in_video=config.post_train_frames,
        test_limit=config.test_dataset_limit,
    )

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

    train_info = train_dataloader.get_dataset_info()
    test_info = test_dataloader.get_dataset_info()
    logger.info("Train dataset: %s", train_info)
    logger.info("Test dataset: %s", test_info)

    if config.compare_tokenizer_checkpoint_path is not None:
        logger.info(
            "Loading comparison tokenizer from %s",
            config.compare_tokenizer_checkpoint_path,
        )
        comparison_tokenizer, _ = load_model_from_checkpoint(
            config.compare_tokenizer_checkpoint_path, device
        )
        compare_tokenizer_rollout_metrics(
            trainable_tokenizer,
            comparison_tokenizer,
            dynamics_model,
            test_dataloader,
            device,
            config,
        )
        if experiment_logger:
            experiment_logger.finish()
        return

    if config.eval_only:
        eval_loss = eval_post_trained_decoder(
            trainable_tokenizer, dynamics_model, test_dataloader,
            device, 0, config, experiment_logger, global_step=0,
        )
        logger.info("Eval-only loss: %.6f", eval_loss)
        if experiment_logger:
            experiment_logger.finish()
        return

    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=1e-4)
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = config.num_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - config.warmup_steps,
        eta_min=config.min_learning_rate,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )

    logger.info(
        "Total optimizer steps: %d, warmup: %d, steps/epoch: %d",
        total_steps, config.warmup_steps, steps_per_epoch,
    )

    early_stopping_state = EarlyStoppingState()

    # Initial eval
    eval_loss = eval_post_trained_decoder(
        trainable_tokenizer, dynamics_model, test_dataloader,
        device, 0, config, experiment_logger, global_step=0,
    )
    logger.info("Initial eval loss: %.6f", eval_loss)

    # Training
    try:
        trainable_tokenizer.train()
        if not config.train_encoder:
            trainable_tokenizer.encoder.eval()
        trainable_tokenizer.fsq.eval()

        for epoch in range(config.num_epochs):
            avg_loss, global_step, should_stop = post_train_epoch(
                trainable_tokenizer, dynamics_model,
                train_dataloader, test_dataloader,
                optimizer, scheduler, device, epoch, config,
                experiment_logger, early_stopping_state,
            )
            logger.info(
                "Epoch %d completed. Avg loss: %.6f", epoch, avg_loss
            )
            if should_stop:
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception:
        logger.exception("Training error")
        raise
    finally:
        if experiment_logger:
            experiment_logger.finish()

    logger.info(
        "Post-training completed. Best eval loss: %.6f",
        early_stopping_state.best_loss,
    )


if __name__ == "__main__":
    config = tyro.cli(PostTrainTokenizerConfig)
    logger.info("Starting post-training... config: %s", config.__dict__)
    main(config)
