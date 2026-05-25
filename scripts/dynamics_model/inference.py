import logging
import random
import subprocess
import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
import traceback
from typing import Literal, Optional

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torchvision.utils as vutils
import tyro

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from dynamics_model.checkpoints import (
    adapt_state_dict_to_model,
    remove_tokenizer_state_dict_entries,
)
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from scripts.dynamics_model.rollout_strategies import (
    ROLLOUT_STRATEGIES,
    rollout_with_strategy,
)
from scripts.video_tokenizer.post_train_tokenizer import (
    PostTrainTokenizerConfig,
    _get_rolling_gt_action_tokens,
    rollout_with_trainable_decoder,
)
from video_tokenization.checkpoints import load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_CONFIG = DynamicsModelTrainingConfig()
VIDEO_FPS = 6
VIDEO_DISPLAY_SCALE = 4


def _load_checkpoint_config(checkpoint_path: Optional[str]) -> dict:
    if checkpoint_path is None:
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return {}
    return checkpoint_config


@dataclass
class InteractiveInferenceArgs:
    tokenizer_checkpoint_path: str
    dynamics_model_checkpoint_path: str
    action_model_checkpoint_path: Optional[str] = BASE_CONFIG.action_model_checkpoint_path
    mode: Literal[
        "interactive",
        "actual_actions_rollout",
        "spam_actions_grid",
        "compare_denoising_steps",
        "compare_rollout_strategies",
        "visualize_denoising_trace",
    ] = "interactive"
    device: str = BASE_CONFIG.device
    num_images_in_video: int = BASE_CONFIG.num_images_in_video
    image_size: int = BASE_CONFIG.image_size
    action_bins: list[int] = field(default_factory=lambda: BASE_CONFIG.action_bins.copy())
    action_d_model: int = BASE_CONFIG.action_d_model
    action_num_transformer_layers: int = BASE_CONFIG.action_num_transformer_layers
    action_num_heads: int = BASE_CONFIG.action_num_heads
    action_latent_dim: int = BASE_CONFIG.action_latent_dim
    dynamics_d_model: int = BASE_CONFIG.dynamics_d_model
    dynamics_num_transformer_layers: int = BASE_CONFIG.dynamics_num_transformer_layers
    dynamics_num_heads: int = BASE_CONFIG.dynamics_num_heads
    batch_size: int = BASE_CONFIG.batch_size
    dataset_type: Literal["pokemon", "atari_pong"] = BASE_CONFIG.dataset_type
    atari_pong_data_dir: Optional[str] = BASE_CONFIG.atari_pong_data_dir
    atari_pong_crop_scoreboard: bool = BASE_CONFIG.atari_pong_crop_scoreboard
    atari_pong_require_full_gameplay: bool = BASE_CONFIG.atari_pong_require_full_gameplay
    frames_dir: str = BASE_CONFIG.frames_dir
    local_cache_dir: Optional[str] = BASE_CONFIG.local_cache_dir
    use_s3: bool = BASE_CONFIG.use_s3
    dataset_train_key: Optional[str] = BASE_CONFIG.dataset_train_key
    sync_from_s3: bool = BASE_CONFIG.sync_from_s3
    output_dir: str = "dynamics_model_results/inference"
    max_steps: int = BASE_CONFIG.rollout_max_denoising_steps
    rollout_total_frames: int = 20
    rollout_seed_frames: Optional[int] = 1
    rollout_num_videos: int = 1
    actual_actions_grid_size: Optional[int] = None
    rollout_pin_first_gt_context: bool = False
    rollout_dataset_limit: int = 1000
    rollout_split: Literal["eval", "train"] = "eval"
    num_workers: int = 0
    spam_grid_rows: int = 4
    spam_grid_cols: int = 4
    spam_action_seed: Optional[int] = None
    spam_actions_override: Optional[list[int]] = None
    compare_denoising_steps: list[int] = field(default_factory=lambda: [5, 15, 25])
    compare_sample_index: Optional[int] = None
    compare_error_gain: float = 16.0
    rollout_strategy_seed: Optional[int] = None
    rollout_strategy_best_of_n: int = 4
    rollout_strategy_keyframe_interval: int = 4
    rollout_strategy_outlier_delta_multiplier: float = 2.5
    triptych_error_gain: float = 5.0


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def _build_config(args: InteractiveInferenceArgs) -> DynamicsModelTrainingConfig:
    config = replace(
        BASE_CONFIG,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=args.dynamics_model_checkpoint_path,
        action_model_checkpoint_path=args.action_model_checkpoint_path,
        device=args.device,
        num_images_in_video=args.num_images_in_video,
        image_size=args.image_size,
        action_bins=args.action_bins,
        action_d_model=args.action_d_model,
        action_num_transformer_layers=args.action_num_transformer_layers,
        action_num_heads=args.action_num_heads,
        action_latent_dim=args.action_latent_dim,
        dynamics_d_model=args.dynamics_d_model,
        dynamics_num_transformer_layers=args.dynamics_num_transformer_layers,
        dynamics_num_heads=args.dynamics_num_heads,
        use_wandb=False,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        atari_pong_data_dir=args.atari_pong_data_dir,
        atari_pong_crop_scoreboard=args.atari_pong_crop_scoreboard,
        atari_pong_require_full_gameplay=args.atari_pong_require_full_gameplay,
        frames_dir=args.frames_dir,
        local_cache_dir=args.local_cache_dir,
        use_s3=args.use_s3,
        dataset_train_key=args.dataset_train_key,
        sync_from_s3=args.sync_from_s3,
    )

    dynamics_checkpoint_config = _load_checkpoint_config(
        args.dynamics_model_checkpoint_path
    )
    action_checkpoint_config = _load_checkpoint_config(args.action_model_checkpoint_path)

    if "action_bins" in action_checkpoint_config:
        config.action_bins = action_checkpoint_config["action_bins"]
    if "action_d_model" in action_checkpoint_config:
        config.action_d_model = action_checkpoint_config["action_d_model"]
    if "action_num_transformer_layers" in action_checkpoint_config:
        config.action_num_transformer_layers = action_checkpoint_config[
            "action_num_transformer_layers"
        ]
    if "action_num_heads" in action_checkpoint_config:
        config.action_num_heads = action_checkpoint_config["action_num_heads"]
    if "action_latent_dim" in action_checkpoint_config:
        config.action_latent_dim = action_checkpoint_config["action_latent_dim"]

    if "dynamics_d_model" in dynamics_checkpoint_config:
        config.dynamics_d_model = dynamics_checkpoint_config["dynamics_d_model"]
    if "dynamics_num_transformer_layers" in dynamics_checkpoint_config:
        config.dynamics_num_transformer_layers = dynamics_checkpoint_config[
            "dynamics_num_transformer_layers"
        ]
    if "dynamics_num_heads" in dynamics_checkpoint_config:
        config.dynamics_num_heads = dynamics_checkpoint_config["dynamics_num_heads"]
    if "num_images_in_video" in dynamics_checkpoint_config:
        config.num_images_in_video = dynamics_checkpoint_config["num_images_in_video"]
    if "image_size" in dynamics_checkpoint_config:
        config.image_size = dynamics_checkpoint_config["image_size"]
    if "patch_size" in dynamics_checkpoint_config:
        config.patch_size = dynamics_checkpoint_config["patch_size"]
    if "frame_spacing" in dynamics_checkpoint_config:
        config.frame_spacing = dynamics_checkpoint_config["frame_spacing"]
    if "dataset_limit" in dynamics_checkpoint_config:
        config.dataset_limit = dynamics_checkpoint_config["dataset_limit"]
    if "num_unique_frames" in dynamics_checkpoint_config:
        config.num_unique_frames = dynamics_checkpoint_config["num_unique_frames"]
    if "dataset_type" in dynamics_checkpoint_config:
        config.dataset_type = dynamics_checkpoint_config["dataset_type"]
    if "atari_pong_data_dir" in dynamics_checkpoint_config:
        config.atari_pong_data_dir = dynamics_checkpoint_config["atari_pong_data_dir"]
    if "atari_pong_crop_scoreboard" in dynamics_checkpoint_config:
        config.atari_pong_crop_scoreboard = dynamics_checkpoint_config["atari_pong_crop_scoreboard"]
    if "atari_pong_require_full_gameplay" in dynamics_checkpoint_config:
        config.atari_pong_require_full_gameplay = dynamics_checkpoint_config["atari_pong_require_full_gameplay"]
    if "predict_action_residuals" in dynamics_checkpoint_config:
        config.predict_action_residuals = dynamics_checkpoint_config[
            "predict_action_residuals"
        ]
    if "action_decoder_loss" in dynamics_checkpoint_config:
        config.action_decoder_loss = dynamics_checkpoint_config["action_decoder_loss"]
    if "action_l2_clip_c" in dynamics_checkpoint_config:
        config.action_l2_clip_c = dynamics_checkpoint_config["action_l2_clip_c"]
    if "dynamics_token_loss" in dynamics_checkpoint_config:
        config.dynamics_token_loss = dynamics_checkpoint_config["dynamics_token_loss"]
    if "dynamics_ce_clip_c" in dynamics_checkpoint_config:
        config.dynamics_ce_clip_c = dynamics_checkpoint_config["dynamics_ce_clip_c"]

    return config


def _load_dynamics_model(
    config: DynamicsModelTrainingConfig, device: torch.device
):
    tokenizer, _ = load_model_from_checkpoint(config.tokenizer_checkpoint_path, device)
    _freeze_module(tokenizer)
    logger.info(
        "Tokenizer decode window from checkpoint: %d frames",
        tokenizer.decoder.num_images_in_video,
    )

    action_model = create_action_model_from_dynamics_config(config).to(device)
    action_model.eval()

    dynamics_model = create_dynamics_model(config, tokenizer, action_model).to(device)

    if config.dynamics_model_checkpoint_path:
        checkpoint = torch.load(config.dynamics_model_checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_to_model(state_dict, dynamics_model)
        state_dict, skipped_tokenizer_keys = remove_tokenizer_state_dict_entries(
            state_dict
        )
        if skipped_tokenizer_keys:
            logger.info(
                "Skipped %d tokenizer keys from dynamics checkpoint so %s remains active",
                skipped_tokenizer_keys,
                config.tokenizer_checkpoint_path,
            )
        missing, unexpected = dynamics_model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.startswith("tokenizer.")]
        if missing:
            logger.warning(
                "Missing %d keys when loading dynamics checkpoint (first 5: %s)",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "Unexpected %d keys when loading dynamics checkpoint (first 5: %s)",
                len(unexpected),
                unexpected[:5],
            )
        logger.info("Loaded dynamics model weights from %s", config.dynamics_model_checkpoint_path)

    dynamics_model.eval()
    return dynamics_model


def _build_test_dataloader(
    config: DynamicsModelTrainingConfig,
    num_frames_in_video: Optional[int] = None,
    dataset_limit: Optional[int] = None,
    num_workers: int = 0,
    split: Literal["eval", "train"] = "eval",
) -> VideoWindowLoader:
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required for inference dataloader")

    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )

    if dataset_limit is not None:
        config = replace(config, dataset_limit=dataset_limit)

    train_dataset, test_dataset = build_datasets(
        config,
        local_cache,
        num_frames_in_video=num_frames_in_video,
    )
    dataset = train_dataset if split == "train" else test_dataset

    test_dataloader = VideoWindowLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=num_workers,
        seed=config.seed,
    )

    logger.info("%s dataloader ready with %d videos", split, len(dataset))
    return test_dataloader


def _normalize_frame(frame: torch.Tensor) -> torch.Tensor:
    """Match training visualizations: treat frames as images in [0, 1]."""
    return frame.detach().float().clamp(0, 1)


def _save_frames_as_video(
    frames: torch.Tensor,
    path: Path,
    fps: int = VIDEO_FPS,
    display_scale: int = VIDEO_DISPLAY_SCALE,
) -> None:
    if frames.dim() == 5:
        frames = frames[0]

    normalized = torch.stack([_normalize_frame(f) for f in frames], dim=0)
    video = (
        normalized.clamp(0, 1)
        .mul(255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    if video.ndim != 4:
        raise ValueError(f"Expected video shape (T, H, W, C), got {video.shape}")
    if display_scale < 1:
        raise ValueError(f"display_scale must be at least 1, got {display_scale}")

    frame_height, frame_width = video.shape[1:3]
    video_filter = (
        f"scale=iw*{display_scale}:ih*{display_scale}:flags=neighbor,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    try:
        subprocess.run(
            ffmpeg_cmd,
            input=video.tobytes(),
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to save browser-compatible MP4s") from exc
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to save video to {path}: {error_message}") from exc


def _save_concatenated_frames(
    frames: torch.Tensor,
    output_dir: Path,
    prefix: str,
    step: int,
) -> tuple[str, str, str]:
    """Save all frames as a single horizontal concatenation.
    
    Args:
        frames: (T, C, H, W) or (1, T, C, H, W) tensor
        output_dir: Directory to save to
        prefix: Filename prefix
        step: Step number for filename
    
    Returns:
        Tuple of (archive_path, session_live_path, live_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if frames.dim() == 5:
        frames = frames[0]  # (T, C, H, W)
    
    # Normalize each frame independently
    normalized = torch.stack([_normalize_frame(f) for f in frames], dim=0)
    
    archive_path = output_dir / f"{prefix}_step_{step:03d}.png"
    session_live_path = output_dir / f"{prefix}_live.png"
    live_path = output_dir / "live.png"
    archive_video_path = output_dir / f"{prefix}_step_{step:03d}.mp4"
    session_live_video_path = output_dir / f"{prefix}_live.mp4"
    live_video_path = output_dir / "live.mp4"
    vutils.save_image(normalized.cpu(), archive_path, nrow=normalized.size(0), padding=2)
    vutils.save_image(
        normalized.cpu(), session_live_path, nrow=normalized.size(0), padding=2
    )
    vutils.save_image(normalized.cpu(), live_path, nrow=normalized.size(0), padding=2)
    _save_frames_as_video(frames, archive_video_path)
    _save_frames_as_video(frames, session_live_video_path)
    _save_frames_as_video(frames, live_video_path)
    return str(archive_path), str(session_live_path), str(live_path)


def _get_actual_next_action(
    model, sampled_video: torch.Tensor, required_context: int
) -> Optional[int]:
    if sampled_video.shape[1] <= required_context:
        return None

    with torch.no_grad():
        encoded_actions = model.action_model.encode(sampled_video)
        action_sequence = model.action_model.get_action_sequence(encoded_actions)

    next_action_idx = required_context - 1
    if action_sequence.shape[1] <= next_action_idx:
        return None
    return int(action_sequence[0, next_action_idx].item())


def _get_action_sequence(model, sampled_video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        encoded_actions = model.action_model.encode(sampled_video)
        return model.action_model.get_action_sequence(encoded_actions).long()


def _rollout_with_context_policy(
    model,
    seed_video: torch.Tensor,
    actions: torch.Tensor,
    max_steps: int,
    pin_first_gt_context: bool,
) -> torch.Tensor:
    if not pin_first_gt_context:
        return model.rollout(seed_video, actions, max_steps=max_steps)

    if seed_video.shape[1] < 1:
        raise ValueError(
            f"Pinned-context rollout requires at least 1 seed frame, got {seed_video.shape[1]}"
        )

    action_sequence = actions.long().to(seed_video.device)
    if action_sequence.dim() == 1:
        if seed_video.shape[0] != 1:
            raise ValueError(
                "Pinned-context rollout expects actions with shape (B, K); got "
                f"a 1D tensor for batch size {seed_video.shape[0]}"
            )
        action_sequence = action_sequence.unsqueeze(0)
    if action_sequence.dim() != 2:
        raise ValueError(
            "Pinned-context rollout expects actions with shape (B, K); got "
            f"{tuple(action_sequence.shape)}"
        )
    if action_sequence.shape[0] != seed_video.shape[0]:
        raise ValueError(
            "Pinned-context rollout actions batch dimension must match seed video: "
            f"{action_sequence.shape[0]} vs {seed_video.shape[0]}"
        )

    max_context = model.num_images_in_video - 1
    pinned_frame = seed_video[:, :1]
    generated = seed_video

    for step in range(action_sequence.shape[1]):
        current_num_frames = generated.shape[1]
        context_size = min(current_num_frames, max_context)
        if current_num_frames <= context_size or context_size <= 1:
            context = generated[:, -context_size:]
        else:
            tail = generated[:, -(context_size - 1) :]
            context = torch.cat([pinned_frame, tail], dim=1)

        next_codes = model.predict_next_frame_codes(
            context,
            action_sequence[:, step],
            max_steps=max_steps,
        )
        next_frame = model.decode_next_frame_with_tokenizer_window(
            generated,
            next_codes,
        )
        generated = torch.cat([generated, next_frame.unsqueeze(1)], dim=1)

    return generated


def _build_actual_actions_grid_video(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    grid_size: int,
    padding: int = 2,
) -> torch.Tensor:
    """Build an ``n x n`` grid where each cell is generated-over-GT."""
    expected_videos = grid_size * grid_size
    if real_videos.shape[0] != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} real videos, got {real_videos.shape[0]}"
        )
    if generated_videos.shape[0] != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} generated videos, "
            f"got {generated_videos.shape[0]}"
        )
    if real_videos.dim() != 5 or generated_videos.dim() != 5:
        raise ValueError(
            "Expected real and generated videos with shape (N,T,C,H,W); "
            f"got real={tuple(real_videos.shape)} "
            f"generated={tuple(generated_videos.shape)}"
        )
    if real_videos.shape != generated_videos.shape:
        raise ValueError(
            "Real and generated video tensors must have identical shape; "
            f"got real={tuple(real_videos.shape)} "
            f"generated={tuple(generated_videos.shape)}"
        )

    real = real_videos.detach().float().clamp(0, 1)
    generated = generated_videos.detach().float().clamp(0, 1)
    separator = torch.ones(
        (
            expected_videos,
            real.shape[1],
            real.shape[2],
            padding,
            real.shape[4],
        ),
        dtype=real.dtype,
        device=real.device,
    )
    paired_cells = torch.cat([generated, separator, real], dim=3)

    grid_frames = []
    for t in range(paired_cells.shape[1]):
        grid_frames.append(
            vutils.make_grid(
                paired_cells[:, t],
                nrow=grid_size,
                padding=padding,
                pad_value=1.0,
            )
        )
    return torch.stack(grid_frames, dim=0)


def _build_rollout_triptych_video(
    real_video: torch.Tensor,
    generated_video: torch.Tensor,
    error_gain: float,
    padding: int = 2,
) -> torch.Tensor:
    """Build GT | generated | abs-error frames for one rollout."""
    if error_gain <= 0:
        raise ValueError(f"triptych_error_gain must be positive, got {error_gain}")
    if real_video.dim() == 5:
        real_video = real_video[0]
    if generated_video.dim() == 5:
        generated_video = generated_video[0]
    if real_video.dim() != 4 or generated_video.dim() != 4:
        raise ValueError(
            "Expected real and generated videos with shape (T,C,H,W) or "
            f"(1,T,C,H,W); got real={tuple(real_video.shape)} "
            f"generated={tuple(generated_video.shape)}"
        )
    if real_video.shape != generated_video.shape:
        raise ValueError(
            "Real and generated video tensors must have identical shape; "
            f"got real={tuple(real_video.shape)} generated={tuple(generated_video.shape)}"
        )

    real = real_video.detach().float().clamp(0, 1)
    generated = generated_video.detach().float().clamp(0, 1)
    error = (generated - real).abs().mul(error_gain).clamp(0, 1)
    separator = torch.ones(
        (real.shape[0], real.shape[1], real.shape[2], padding),
        dtype=real.dtype,
        device=real.device,
    )
    return torch.cat([real, separator, generated, separator, error], dim=3)


def _build_rollout_triptych_grid_video(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    grid_size: int,
    error_gain: float,
    padding: int = 2,
) -> torch.Tensor:
    """Build an ``n x n`` grid where each cell is GT | generated | abs-error."""
    expected_videos = grid_size * grid_size
    if real_videos.shape[0] != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} real videos, got {real_videos.shape[0]}"
        )
    if generated_videos.shape[0] != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} generated videos, got {generated_videos.shape[0]}"
        )
    if real_videos.shape != generated_videos.shape:
        raise ValueError(
            "Real and generated video tensors must have identical shape; "
            f"got real={tuple(real_videos.shape)} generated={tuple(generated_videos.shape)}"
        )

    triptych_cells = []
    for video_idx in range(expected_videos):
        triptych_cells.append(
            _build_rollout_triptych_video(
                real_videos[video_idx],
                generated_videos[video_idx],
                error_gain,
                padding=padding,
            )
        )
    triptych_videos = torch.stack(triptych_cells, dim=0)

    grid_frames = []
    for t in range(triptych_videos.shape[1]):
        grid_frames.append(
            vutils.make_grid(
                triptych_videos[:, t],
                nrow=grid_size,
                padding=padding,
                pad_value=1.0,
            )
        )
    return torch.stack(grid_frames, dim=0)


def _actual_actions_rollout(
    model,
    test_dataloader: VideoWindowLoader,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    if args.rollout_total_frames < 2:
        raise ValueError("rollout_total_frames must be at least 2")
    rollout_seed_frames = (
        args.rollout_seed_frames
        if args.rollout_seed_frames is not None
        else model.num_images_in_video
    )
    if not 1 <= rollout_seed_frames < args.rollout_total_frames:
        raise ValueError(
            "rollout_seed_frames must be at least 1 and less than rollout_total_frames"
        )
    if args.rollout_num_videos < 1:
        raise ValueError("rollout_num_videos must be at least 1")
    if args.actual_actions_grid_size is not None and args.actual_actions_grid_size < 1:
        raise ValueError("actual_actions_grid_size must be at least 1 when set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_size = args.actual_actions_grid_size
    target_videos = grid_size * grid_size if grid_size is not None else args.rollout_num_videos
    expected_actions = args.rollout_total_frames - rollout_seed_frames
    saved_count = 0
    loader_iter = iter(test_dataloader)
    grid_real_videos: list[torch.Tensor] = []
    grid_generated_videos: list[torch.Tensor] = []
    grid_action_lists: list[list[int]] = []

    while saved_count < target_videos:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_dataloader)
            batch = next(loader_iter)

        if batch.dim() != 5:
            raise ValueError("Expected batch shape (B, T, C, H, W)")

        sample_indices = torch.randperm(batch.size(0)).tolist()
        for idx in sample_indices:
            if saved_count >= target_videos:
                break

            real_video = batch[idx : idx + 1].to(device)
            if real_video.shape[1] < args.rollout_total_frames:
                raise ValueError(
                    f"Expected at least {args.rollout_total_frames} frames, "
                    f"got {real_video.shape[1]}"
                )

            real_video = real_video[:, : args.rollout_total_frames]
            action_sequence = _get_action_sequence(model, real_video)
            rollout_actions = action_sequence[
                :,
                rollout_seed_frames - 1 : args.rollout_total_frames - 1,
            ]
            if rollout_actions.shape[1] != expected_actions:
                raise ValueError(
                    f"Expected {expected_actions} rollout actions, got "
                    f"{rollout_actions.shape[1]}"
                )

            seed_video = real_video[:, :rollout_seed_frames]
            with torch.no_grad():
                generated_video = _rollout_with_context_policy(
                    model,
                    seed_video,
                    rollout_actions,
                    args.max_steps,
                    args.rollout_pin_first_gt_context,
                )

            if grid_size is not None:
                grid_real_videos.append(real_video[0].detach().cpu())
                grid_generated_videos.append(generated_video[0].detach().cpu())
                grid_action_lists.append(rollout_actions[0].detach().cpu().tolist())
                print(
                    "Actual-actions grid sample complete: "
                    f"video={saved_count + 1}/{target_videos} "
                    f"batch_sample_idx={idx} seed_frames={rollout_seed_frames} "
                    f"total_frames={args.rollout_total_frames} "
                    f"actions={grid_action_lists[-1]}"
                )
                saved_count += 1
                continue

            if args.rollout_num_videos == 1:
                real_prefix = "actual_actions_rollout_real"
                generated_prefix = "actual_actions_rollout_generated"
            else:
                real_prefix = f"actual_actions_rollout_{saved_count:03d}_real"
                generated_prefix = f"actual_actions_rollout_{saved_count:03d}_generated"

            real_path, real_live_path, _ = _save_concatenated_frames(
                real_video,
                output_dir,
                real_prefix,
                0,
            )
            generated_path, generated_live_path, live_path = _save_concatenated_frames(
                generated_video,
                output_dir,
                generated_prefix,
                0,
            )
            triptych_video = _build_rollout_triptych_video(
                real_video,
                generated_video,
                args.triptych_error_gain,
            )
            triptych_path = output_dir / f"{generated_prefix}_triptych.mp4"
            _save_frames_as_video(triptych_video, triptych_path, display_scale=2)

            print(
                "Actual-actions rollout complete: "
                f"video={saved_count + 1}/{args.rollout_num_videos} "
                f"batch_sample_idx={idx} seed_frames={rollout_seed_frames} "
                f"total_frames={args.rollout_total_frames} "
                f"actions={rollout_actions[0].detach().cpu().tolist()} "
                f"generated={generated_path} generated_live={generated_live_path} "
                f"real={real_path} real_live={real_live_path} live={live_path} "
                f"triptych={triptych_path}"
            )
            saved_count += 1

    if grid_size is not None:
        real_videos = torch.stack(grid_real_videos, dim=0)
        generated_videos = torch.stack(grid_generated_videos, dim=0)
        grid_video = _build_actual_actions_grid_video(
            real_videos,
            generated_videos,
            grid_size,
        )
        grid_path = output_dir / "actual_actions_rollout_grid.mp4"
        _save_frames_as_video(grid_video, grid_path, display_scale=2)
        triptych_grid_video = _build_rollout_triptych_grid_video(
            real_videos,
            generated_videos,
            grid_size,
            args.triptych_error_gain,
        )
        triptych_grid_path = output_dir / "actual_actions_rollout_triptych_grid.mp4"
        _save_frames_as_video(triptych_grid_video, triptych_grid_path, display_scale=2)
        print(
            "Actual-actions rollout grid complete: "
            f"grid={grid_size}x{grid_size} seed_frames={rollout_seed_frames} "
            f"total_frames={args.rollout_total_frames} grid_path={grid_path} "
            f"triptych_grid_path={triptych_grid_path}"
        )
        print(
            "Each standard grid cell shows generated rollout above ground truth; "
            "each triptych grid cell shows GT | generated | abs error."
        )
        for i, actions in enumerate(grid_action_lists):
            row, col = divmod(i, grid_size)
            print(f"  cell[r={row} c={col}] actions={actions}")


def _build_spam_grid_video(
    gt_rollout: torch.Tensor,
    spam_rollouts: torch.Tensor,
    rows: int,
    cols: int,
    padding: int = 2,
) -> torch.Tensor:
    """Stack a GT rollout on top of a ``rows x cols`` spam-action grid.

    Args:
        gt_rollout: ``(T, C, H, W)`` ground-truth-actions rollout.
        spam_rollouts: ``(rows * cols, T, C, H, W)`` random-action rollouts.
        rows, cols: bottom grid layout.
        padding: pixels between grid cells (and between top/bottom strips).

    Returns:
        ``(T, C, total_H, total_W)`` tensor in ``[0, 1]`` ready for ffmpeg.
    """
    if spam_rollouts.shape[0] != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} spam rollouts, got {spam_rollouts.shape[0]}"
        )
    if spam_rollouts.dim() != 5 or gt_rollout.dim() != 4:
        raise ValueError(
            "Expected gt_rollout (T,C,H,W) and spam_rollouts (N,T,C,H,W); "
            f"got gt={tuple(gt_rollout.shape)} spam={tuple(spam_rollouts.shape)}"
        )

    n_frames = spam_rollouts.shape[1]
    cell_h = spam_rollouts.shape[3]

    bottom_frames = []
    for t in range(n_frames):
        cells = spam_rollouts[:, t].detach().float().clamp(0, 1)
        bottom_frames.append(
            vutils.make_grid(cells, nrow=cols, padding=padding, pad_value=1.0)
        )
    bottom = torch.stack(bottom_frames, dim=0)
    bottom_w = bottom.shape[-1]

    gt = gt_rollout.detach().float().clamp(0, 1)
    gt_resized = torch.nn.functional.interpolate(
        gt, size=(cell_h, bottom_w), mode="nearest"
    )

    if padding > 0:
        separator = torch.ones(
            (n_frames, gt_resized.shape[1], padding, bottom_w),
            dtype=gt_resized.dtype,
            device=gt_resized.device,
        )
        return torch.cat([gt_resized, separator, bottom], dim=2)
    return torch.cat([gt_resized, bottom], dim=2)


def _spam_actions_grid(
    model,
    test_dataloader: VideoWindowLoader,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    """Render one ground-truth rollout above a grid of spam-action rollouts.

    Picks one window from the loader, extracts the model's predicted
    ground-truth action sequence (same path as ``actual_actions_rollout``),
    then for ``rows * cols`` cells samples a single random action from the
    action codebook and repeats it for every rollout step. All rollouts
    (1 GT + ``rows * cols`` spam) are batched into a single ``model.rollout``
    call and assembled into one mp4 via ``_build_spam_grid_video``.
    """
    if args.rollout_total_frames < 2:
        raise ValueError("rollout_total_frames must be at least 2")
    rollout_seed_frames = (
        args.rollout_seed_frames
        if args.rollout_seed_frames is not None
        else model.num_images_in_video
    )
    if not 1 <= rollout_seed_frames < args.rollout_total_frames:
        raise ValueError(
            "rollout_seed_frames must be at least 1 and less than rollout_total_frames"
        )
    if args.spam_grid_rows < 1 or args.spam_grid_cols < 1:
        raise ValueError("spam_grid_rows and spam_grid_cols must be >= 1")

    rows = args.spam_grid_rows
    cols = args.spam_grid_cols
    n_random = rows * cols
    expected_actions = args.rollout_total_frames - rollout_seed_frames

    batch = next(iter(test_dataloader))
    if batch.dim() != 5:
        raise ValueError("Expected batch shape (B, T, C, H, W)")
    if batch.shape[1] < args.rollout_total_frames:
        raise ValueError(
            f"Loader returned {batch.shape[1]} frames; need at least "
            f"{args.rollout_total_frames}"
        )
    sample_idx = random.randint(0, batch.size(0) - 1)
    real_video = batch[sample_idx : sample_idx + 1, : args.rollout_total_frames].to(
        device
    )

    action_sequence = _get_action_sequence(model, real_video)
    gt_actions = action_sequence[
        :, rollout_seed_frames - 1 : args.rollout_total_frames - 1
    ]
    if gt_actions.shape[1] != expected_actions:
        raise ValueError(
            f"Expected {expected_actions} GT actions, got {gt_actions.shape[1]}"
        )

    vocab_size = model.action_model.action_vocab_size
    rng = random.Random(args.spam_action_seed)
    if args.spam_actions_override is not None:
        if len(args.spam_actions_override) != n_random:
            raise ValueError(
                f"spam_actions_override has {len(args.spam_actions_override)} "
                f"entries but grid has {n_random} cells"
            )
        spam_choices = list(args.spam_actions_override)
    else:
        spam_choices = [rng.randrange(vocab_size) for _ in range(n_random)]
    spam_actions = torch.tensor(
        [[a] * expected_actions for a in spam_choices],
        dtype=torch.long,
        device=device,
    )

    all_actions = torch.cat([gt_actions, spam_actions], dim=0)
    seed_video = real_video[:, :rollout_seed_frames]
    seed_video = seed_video.expand(all_actions.shape[0], -1, -1, -1, -1).contiguous()

    with torch.no_grad():
        all_rollouts = _rollout_with_context_policy(
            model,
            seed_video,
            all_actions,
            args.max_steps,
            args.rollout_pin_first_gt_context,
        )

    if all_rollouts.shape[1] != args.rollout_total_frames:
        raise ValueError(
            f"Rollout returned {all_rollouts.shape[1]} frames, expected "
            f"{args.rollout_total_frames}"
        )

    gt_rollout = all_rollouts[0]
    spam_rollouts = all_rollouts[1:]
    grid_video = _build_spam_grid_video(gt_rollout, spam_rollouts, rows, cols)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "spam_actions_grid.mp4"
    _save_frames_as_video(grid_video, out_path, display_scale=2)

    real_strip_path, _, _ = _save_concatenated_frames(
        real_video, output_dir, "spam_actions_grid_real_strip", 0
    )

    gt_action_list = gt_actions[0].detach().cpu().tolist()
    print(
        "Spam-actions grid complete: "
        f"sample_idx={sample_idx} seed_frames={rollout_seed_frames} "
        f"total_frames={args.rollout_total_frames} "
        f"action_vocab_size={vocab_size} "
        f"grid={rows}x{cols} grid_path={out_path} "
        f"real_strip={real_strip_path} "
        f"gt_actions={gt_action_list}"
    )
    print(f"Cell action map ({rows}x{cols}, row-major from top-left of bottom grid):")
    for i, action_id in enumerate(spam_choices):
        row, col = divmod(i, cols)
        print(f"  cell[r={row} c={col}] action={action_id}")


def _add_label_strip(
    frame: torch.Tensor,
    label: str,
    label_width: int = 96,
) -> torch.Tensor:
    frame = _normalize_frame(frame)
    if frame.dim() != 3:
        raise ValueError(f"Expected frame shape (C, H, W), got {tuple(frame.shape)}")
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    if frame.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {frame.shape[0]}")

    frame_uint8 = (
        frame.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    )
    image = Image.fromarray(frame_uint8, mode="RGB")
    canvas = Image.new("RGB", (image.width + label_width, image.height), "white")
    canvas.paste(image, (label_width, 0))
    draw = ImageDraw.Draw(canvas)
    text_y = max(0, (image.height - 10) // 2)
    draw.text((4, text_y), label, fill="black")

    labeled = torch.tensor(
        list(canvas.getdata()),
        dtype=torch.uint8,
    ).view(canvas.height, canvas.width, 3)
    return labeled.permute(2, 0, 1).float().div(255)


def _build_labeled_rows_video(rows: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    if not rows:
        raise ValueError("Expected at least one video row")

    n_frames = rows[0][1].shape[0]
    for label, video in rows:
        if video.dim() != 4:
            raise ValueError(
                f"Expected video row '{label}' to have shape (T,C,H,W), "
                f"got {tuple(video.shape)}"
            )
        if video.shape[0] != n_frames:
            raise ValueError(
                f"Video row '{label}' has {video.shape[0]} frames; expected {n_frames}"
            )

    comparison_frames = []
    separator_width = rows[0][1].shape[-1] + 96
    for t in range(n_frames):
        labeled_rows = []
        for label, video in rows:
            labeled_rows.append(_add_label_strip(video[t], label))
            separator = torch.ones(
                (3, 2, labeled_rows[-1].shape[-1]),
                dtype=labeled_rows[-1].dtype,
            )
            labeled_rows.append(separator)
        if labeled_rows:
            labeled_rows.pop()
        frame = torch.cat(labeled_rows, dim=1)
        if frame.shape[-1] != separator_width:
            raise ValueError(
                f"Unexpected labeled frame width {frame.shape[-1]}, "
                f"expected {separator_width}"
            )
        comparison_frames.append(frame)

    return torch.stack(comparison_frames, dim=0)


def _build_raw_error_rows_video(
    real_video: torch.Tensor,
    generated_by_step: dict[int, torch.Tensor],
    error_gain: float,
) -> torch.Tensor:
    if error_gain <= 0:
        raise ValueError(f"error_gain must be positive, got {error_gain}")

    rows: list[tuple[str, torch.Tensor]] = []
    for step_count in sorted(generated_by_step):
        generated = generated_by_step[step_count].detach().float().clamp(0, 1)
        error = (generated - real_video).abs().mul(error_gain).clamp(0, 1)
        separator = torch.ones(
            (generated.shape[0], generated.shape[1], generated.shape[2], 2),
            dtype=generated.dtype,
        )
        raw_and_error = torch.cat([generated, separator, error], dim=3)
        rows.append((f"{step_count} raw | err x{error_gain:g}", raw_and_error))

    return _build_labeled_rows_video(rows)


def _plot_denoising_step_metrics(
    real_video: torch.Tensor,
    generated_by_step: dict[int, torch.Tensor],
    rollout_seed_frames: int,
    output_path: Path,
) -> None:
    frame_numbers = list(range(rollout_seed_frames, real_video.shape[0]))
    mse_by_step: dict[int, list[float]] = {}
    psnr_by_step: dict[int, list[float]] = {}

    for step_count, generated_video in generated_by_step.items():
        generated_frames = generated_video[rollout_seed_frames:]
        real_frames = real_video[rollout_seed_frames:]
        mse = (generated_frames - real_frames).pow(2).mean(dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        mse_by_step[step_count] = mse.detach().cpu().tolist()
        psnr_by_step[step_count] = psnr.detach().cpu().tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for step_count in sorted(generated_by_step):
        axes[0].plot(
            frame_numbers,
            mse_by_step[step_count],
            marker="o",
            label=f"{step_count} steps",
        )
        axes[1].plot(
            frame_numbers,
            psnr_by_step[step_count],
            marker="o",
            label=f"{step_count} steps",
        )

    axes[0].set_ylabel("MSE vs GT")
    axes[0].set_title("Rollout error by denoising steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_ylabel("PSNR vs GT")
    axes[1].set_xlabel("Frame index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_denoising_step_metrics_csv(
    real_video: torch.Tensor,
    generated_by_step: dict[int, torch.Tensor],
    rollout_seed_frames: int,
    output_path: Path,
) -> dict[int, dict[str, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[int, dict[str, float]] = {}
    rows: list[dict[str, float | int]] = []

    for step_count in sorted(generated_by_step):
        generated_video = generated_by_step[step_count]
        generated_frames = generated_video[rollout_seed_frames:]
        real_frames = real_video[rollout_seed_frames:]
        mse = (generated_frames - real_frames).pow(2).mean(dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        summary[step_count] = {
            "mean_mse": float(mse.mean().item()),
            "final_mse": float(mse[-1].item()),
            "mean_psnr": float(psnr.mean().item()),
            "final_psnr": float(psnr[-1].item()),
        }
        for frame_idx, frame_mse, frame_psnr in zip(
            range(rollout_seed_frames, real_video.shape[0]),
            mse.tolist(),
            psnr.tolist(),
        ):
            rows.append(
                {
                    "denoising_steps": step_count,
                    "frame_idx": frame_idx,
                    "mse": frame_mse,
                    "psnr": frame_psnr,
                }
            )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["denoising_steps", "frame_idx", "mse", "psnr"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return summary


def _frame_metrics_after_seed(
    real_video: torch.Tensor,
    generated_video: torch.Tensor,
    rollout_seed_frames: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    frame_numbers = list(range(rollout_seed_frames, real_video.shape[0]))
    generated_frames = generated_video[rollout_seed_frames:]
    real_frames = real_video[rollout_seed_frames:]
    mse = (generated_frames - real_frames).pow(2).mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return frame_numbers, mse, psnr


def _plot_denoising_seed_compare_metrics(
    real_video: torch.Tensor,
    generated_by_condition: dict[str, dict[int, torch.Tensor]],
    seed_frames_by_condition: dict[str, int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for condition, generated_by_step in generated_by_condition.items():
        seed_frames = seed_frames_by_condition[condition]
        linestyle = "-" if seed_frames == 1 else "--"
        for step_count in sorted(generated_by_step):
            frame_numbers, mse, psnr = _frame_metrics_after_seed(
                real_video,
                generated_by_step[step_count],
                seed_frames,
            )
            label = f"{step_count} steps, {condition}"
            axes[0].plot(
                frame_numbers,
                mse.detach().cpu().tolist(),
                marker="o",
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                frame_numbers,
                psnr.detach().cpu().tolist(),
                marker="o",
                linestyle=linestyle,
                label=label,
            )

    axes[0].set_ylabel("MSE vs GT (log)")
    axes[0].set_yscale("log")
    axes[0].set_title("Denoising-step rollout error: seeded context vs true AR")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("PSNR vs GT")
    axes[1].set_xlabel("Frame index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_denoising_seed_compare_summary_csv(
    real_video: torch.Tensor,
    generated_by_condition: dict[str, dict[int, torch.Tensor]],
    seed_frames_by_condition: dict[str, int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for condition, generated_by_step in generated_by_condition.items():
        seed_frames = seed_frames_by_condition[condition]
        for step_count in sorted(generated_by_step):
            _frame_numbers, mse, psnr = _frame_metrics_after_seed(
                real_video,
                generated_by_step[step_count],
                seed_frames,
            )
            rows.append(
                {
                    "condition": condition,
                    "denoising_steps": step_count,
                    "mean_mse": float(mse.mean().item()),
                    "final_mse": float(mse[-1].item()),
                    "mean_psnr": float(psnr.mean().item()),
                    "final_psnr": float(psnr[-1].item()),
                }
            )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "condition",
                "denoising_steps",
                "mean_mse",
                "final_mse",
                "mean_psnr",
                "final_psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _post_train_style_rollout(
    model,
    real_video: torch.Tensor,
    seed_frames: int,
    total_frames: int,
    max_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Mirror post_train_rollout_mp4.py's rolling decoder/action-token path."""
    post_train_config = PostTrainTokenizerConfig(
        post_train_frames=total_frames,
        dynamics_context_frames=model.num_images_in_video,
        rollout_seed_frames=seed_frames,
        max_denoising_steps=max_steps,
        device=str(device),
    )
    use_amp = device.type == "cuda"
    rollout_decoded, _recon_decoded, _gt_future = rollout_with_trainable_decoder(
        model,
        model.tokenizer,
        real_video[:, :total_frames],
        post_train_config,
        use_amp,
    )
    return torch.cat(
        [real_video[:, :seed_frames], rollout_decoded],
        dim=1,
    )


def _committed_patch_mask_to_frame(
    committed_mask: torch.Tensor,
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    patch_rows = image_height // patch_height
    patch_cols = image_width // patch_width
    expected_patches = patch_rows * patch_cols
    if committed_mask.numel() != expected_patches:
        raise ValueError(
            "Committed mask patch count does not match image geometry: "
            f"mask={committed_mask.numel()} patches={expected_patches}"
        )
    mask = committed_mask.float().view(1, 1, patch_rows, patch_cols)
    mask = torch.nn.functional.interpolate(
        mask,
        size=(image_height, image_width),
        mode="nearest",
    )[0]
    return mask.repeat(3, 1, 1)


def _apply_patch_visibility_mask(
    frame: torch.Tensor,
    committed_mask: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
    hidden_value: float = 1.0,
) -> torch.Tensor:
    mask_frame = _committed_patch_mask_to_frame(
        committed_mask,
        image_height=image_height,
        image_width=image_width,
        patch_height=patch_height,
        patch_width=patch_width,
    )
    visible = frame.detach().float().cpu().clamp(0, 1)
    hidden = torch.full_like(visible, hidden_value)
    return torch.where(mask_frame.bool(), visible, hidden)


@torch.no_grad()
def _predict_next_frame_denoising_trace(
    model,
    generated_pixels: torch.Tensor,
    context_pixels: torch.Tensor,
    action: torch.Tensor,
    context_actions: torch.Tensor | None,
    max_steps: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[float]]:
    max_context = model.num_images_in_video - 1
    context_window = context_pixels[:, -max_context:]
    placeholder = context_window[:, -1:].clone()
    inference_window = torch.cat([context_window, placeholder], dim=1)

    targets = model.tokenizer.quantized_value_to_codes(
        model.tokenizer.encode(inference_window)
    ).long()
    batch_size, _, num_patches = targets.shape
    fallback_next_codes = targets[:, -1, :].clone()

    action_token = action.long().to(targets.device)
    if action_token.dim() == 1:
        action_token = action_token.unsqueeze(1)

    targets[:, -1, :] = model.tokenizer.get_mask_token_idx()
    expected_context = inference_window.shape[1] - 2
    if context_actions is not None:
        context_action_tokens = context_actions.long().to(targets.device)
        if context_action_tokens.shape != (batch_size, expected_context):
            raise ValueError(
                f"context_actions must have shape (B, {expected_context}), "
                f"got {tuple(context_action_tokens.shape)}"
            )
    elif expected_context > 0:
        action_video_encoded = model.action_model.encode(inference_window[:, :-1])
        context_action_tokens = model.action_model.get_action_sequence(
            action_video_encoded
        ).long()
    else:
        context_action_tokens = torch.empty(
            batch_size, 0, dtype=torch.long, device=targets.device
        )

    action_tokens = torch.cat([context_action_tokens, action_token], dim=1)
    action_embeddings = model.action_embedding(action_tokens.long())

    trace_frames: list[torch.Tensor] = []
    trace_committed_only_frames: list[torch.Tensor] = []
    trace_masks: list[torch.Tensor] = []
    committed_fractions: list[float] = []
    mask_locations = torch.full(
        (batch_size, num_patches), True, dtype=torch.bool, device=targets.device
    )
    for step in range(max_steps):
        x = model.tokenizer_embedding(targets.long())
        x[:, :-1, :] += action_embeddings.unsqueeze(2)
        x = model.decoder(x)
        logits = model.vocab_head(x)

        if mask_locations.any():
            ratio_remaining = model.cosine_scheduler(
                max_steps, step + 1, targets.device
            )
            target_still_masked = int(num_patches * ratio_remaining)
            current_masked = int(mask_locations.sum(dim=1).amax().item())
            tokens_to_update = max(0, current_masked - target_still_masked)

            probs = model.softmax(logits[:, -1])
            mask_flat = mask_locations.view(-1)
            probs_flat = probs.view(-1, probs.size(-1))
            samples_flat = torch.full(
                (batch_size * num_patches,),
                -100,
                device=targets.device,
                dtype=torch.long,
            )
            if mask_flat.any():
                probs_masked = probs_flat[mask_flat]
                probs_masked = probs_masked / (
                    probs_masked.sum(-1, keepdim=True) + 1e-9
                )
                sampled_masked = torch.multinomial(
                    probs_masked, 1, replacement=False
                ).squeeze(-1)
                samples_flat[mask_flat] = sampled_masked

            samples = samples_flat.view(batch_size, num_patches)
            sampled_scores = probs.gather(
                -1, samples.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            sampled_scores[samples < 0] = -1

            if tokens_to_update > 0:
                _, top_positions = torch.topk(
                    sampled_scores, tokens_to_update, dim=-1
                )
                top_tokens = samples.gather(1, top_positions)
                targets_last = targets[:, -1, :]
                targets_last = targets_last.scatter(1, top_positions, top_tokens)
                targets[:, -1, :] = targets_last
                mask_locations = mask_locations.scatter(
                    1, top_positions, torch.zeros_like(top_positions, dtype=torch.bool)
                )

        committed_mask = ~mask_locations
        visual_codes = fallback_next_codes.clone()
        targets_last = targets[:, -1, :]
        visual_codes[committed_mask] = targets_last[committed_mask]
        decoded_frame = model.decode_next_frame_with_tokenizer_window(
            generated_pixels,
            visual_codes,
        )
        decoded_frame_cpu = decoded_frame[0].detach().cpu()
        committed_mask_cpu = committed_mask[0].detach().cpu()
        trace_frames.append(decoded_frame_cpu)
        trace_committed_only_frames.append(
            _apply_patch_visibility_mask(
                decoded_frame_cpu,
                committed_mask_cpu,
                image_height=generated_pixels.shape[-2],
                image_width=generated_pixels.shape[-1],
                patch_height=model.tokenizer.encoder.patch_height,
                patch_width=model.tokenizer.encoder.patch_width,
            )
        )
        trace_masks.append(committed_mask_cpu)
        committed_fractions.append(float(committed_mask.float().mean().item()))

    return trace_frames, trace_committed_only_frames, trace_masks, committed_fractions


def _visualize_denoising_trace(
    model,
    test_dataloader: VideoWindowLoader,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    rollout_seed_frames = (
        args.rollout_seed_frames
        if args.rollout_seed_frames is not None
        else model.num_images_in_video
    )
    if rollout_seed_frames < 1:
        raise ValueError("rollout_seed_frames must be at least 1")

    batch = next(iter(test_dataloader))
    if batch.dim() != 5:
        raise ValueError("Expected batch shape (B, T, C, H, W)")
    target_frame_idx = rollout_seed_frames
    if batch.shape[1] <= target_frame_idx:
        raise ValueError(
            f"Loader returned {batch.shape[1]} frames; need at least "
            f"{target_frame_idx + 1}"
        )

    if args.compare_sample_index is None:
        sample_idx = random.randint(0, batch.size(0) - 1)
    else:
        if not 0 <= args.compare_sample_index < batch.size(0):
            raise ValueError(
                f"compare_sample_index must be in [0, {batch.size(0) - 1}], "
                f"got {args.compare_sample_index}"
            )
        sample_idx = args.compare_sample_index

    real_video = batch[
        sample_idx : sample_idx + 1, : target_frame_idx + 1
    ].to(device)
    generated_pixels = real_video[:, :rollout_seed_frames]
    ctx_size = min(generated_pixels.shape[1], model.num_images_in_video - 1)
    context_pixels = generated_pixels[:, -ctx_size:]
    context_actions, next_action = _get_rolling_gt_action_tokens(
        model,
        real_video,
        target_frame_idx,
        ctx_size,
    )

    trace_frames, trace_committed_only_frames, trace_masks, committed_fractions = (
        _predict_next_frame_denoising_trace(
            model,
            generated_pixels,
            context_pixels,
            next_action,
            context_actions,
            args.max_steps,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_frames = []
    labeled_committed_only_frames = []
    labeled_masks = []
    image_height = real_video.shape[-2]
    image_width = real_video.shape[-1]
    patch_height = model.tokenizer.encoder.patch_height
    patch_width = model.tokenizer.encoder.patch_width
    for step_idx, (frame, mask, committed_fraction) in enumerate(
        zip(trace_frames, trace_masks, committed_fractions),
        start=1,
    ):
        label = f"step {step_idx}/{args.max_steps}\n{committed_fraction:.0%} filled"
        labeled_frames.append(_add_label_strip(frame, label))
        labeled_committed_only_frames.append(
            _add_label_strip(trace_committed_only_frames[step_idx - 1], label)
        )
        mask_frame = _committed_patch_mask_to_frame(
            mask,
            image_height=image_height,
            image_width=image_width,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        labeled_masks.append(_add_label_strip(mask_frame, label))

    gt_labeled = _add_label_strip(real_video[0, target_frame_idx].detach().cpu(), "GT target")
    prev_labeled = _add_label_strip(generated_pixels[0, -1].detach().cpu(), "prev frame")
    frame_grid = torch.stack([prev_labeled, *labeled_frames, gt_labeled], dim=0)
    committed_only_grid = torch.stack(
        [prev_labeled, *labeled_committed_only_frames, gt_labeled],
        dim=0,
    )
    mask_grid = torch.stack(labeled_masks, dim=0)
    frame_path = output_dir / "denoising_trace_frames.png"
    committed_only_path = output_dir / "denoising_trace_committed_only.png"
    mask_path = output_dir / "denoising_trace_committed_masks.png"
    vutils.save_image(frame_grid, frame_path, nrow=min(4, frame_grid.shape[0]), padding=2)
    vutils.save_image(
        committed_only_grid,
        committed_only_path,
        nrow=min(4, committed_only_grid.shape[0]),
        padding=2,
    )
    vutils.save_image(mask_grid, mask_path, nrow=min(5, mask_grid.shape[0]), padding=2)
    _save_frames_as_video(
        torch.stack(trace_frames, dim=0),
        output_dir / "denoising_trace_frames.mp4",
        display_scale=4,
    )
    _save_frames_as_video(
        torch.stack(trace_committed_only_frames, dim=0),
        output_dir / "denoising_trace_committed_only.mp4",
        display_scale=4,
    )
    print(
        "Denoising trace complete: "
        f"sample_idx={sample_idx} seed_frames={rollout_seed_frames} "
        f"target_frame_idx={target_frame_idx} action={int(next_action[0].item())} "
        f"steps={args.max_steps} frames={frame_path} "
        f"committed_only={committed_only_path} masks={mask_path}"
    )


def _compare_denoising_steps(
    model,
    test_dataloader: VideoWindowLoader,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    if args.rollout_total_frames < 2:
        raise ValueError("rollout_total_frames must be at least 2")
    rollout_seed_frames = (
        args.rollout_seed_frames
        if args.rollout_seed_frames is not None
        else model.num_images_in_video
    )
    if not 1 <= rollout_seed_frames < args.rollout_total_frames:
        raise ValueError(
            "rollout_seed_frames must be at least 1 and less than rollout_total_frames"
        )
    if not args.compare_denoising_steps:
        raise ValueError("compare_denoising_steps must include at least one value")
    if any(step_count < 1 for step_count in args.compare_denoising_steps):
        raise ValueError("All compare_denoising_steps values must be at least 1")

    batch = next(iter(test_dataloader))
    if batch.dim() != 5:
        raise ValueError("Expected batch shape (B, T, C, H, W)")
    if batch.shape[1] < args.rollout_total_frames:
        raise ValueError(
            f"Loader returned {batch.shape[1]} frames; need at least "
            f"{args.rollout_total_frames}"
        )

    if args.compare_sample_index is None:
        sample_idx = random.randint(0, batch.size(0) - 1)
    else:
        if not 0 <= args.compare_sample_index < batch.size(0):
            raise ValueError(
                f"compare_sample_index must be in [0, {batch.size(0) - 1}], "
                f"got {args.compare_sample_index}"
            )
        sample_idx = args.compare_sample_index

    real_video = batch[
        sample_idx : sample_idx + 1, : args.rollout_total_frames
    ].to(device)
    seed_conditions = [(f"{rollout_seed_frames} seed frames", rollout_seed_frames)]
    if rollout_seed_frames != 1:
        seed_conditions.append(("1 seed frame (true AR)", 1))

    generated_by_condition: dict[str, dict[int, torch.Tensor]] = {}
    seed_frames_by_condition: dict[str, int] = {}
    for condition, seed_frames in seed_conditions:
        generated_by_step: dict[int, torch.Tensor] = {}
        for step_count in args.compare_denoising_steps:
            with torch.no_grad():
                generated = _post_train_style_rollout(
                    model,
                    real_video,
                    seed_frames,
                    args.rollout_total_frames,
                    max_steps=step_count,
                    device=device,
                )
            generated_by_step[step_count] = generated[0].detach().cpu()
        generated_by_condition[condition] = generated_by_step
        seed_frames_by_condition[condition] = seed_frames

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_cpu = real_video[0].detach().cpu()
    _save_frames_as_video(real_cpu, output_dir / "denoising_steps_gt.mp4")
    generated_by_step = generated_by_condition[seed_conditions[0][0]]
    for step_count, generated in generated_by_step.items():
        _save_frames_as_video(
            generated,
            output_dir / f"denoising_steps_{step_count:03d}.mp4",
        )

    plot_path = output_dir / "denoising_steps_metrics.png"
    _plot_denoising_step_metrics(
        real_cpu,
        generated_by_step,
        rollout_seed_frames,
        plot_path,
    )
    csv_path = output_dir / "denoising_steps_metrics.csv"
    metrics_summary = _save_denoising_step_metrics_csv(
        real_cpu,
        generated_by_step,
        rollout_seed_frames,
        csv_path,
    )
    seed_compare_plot_path = output_dir / "denoising_steps_seed_compare.png"
    _plot_denoising_seed_compare_metrics(
        real_cpu,
        generated_by_condition,
        seed_frames_by_condition,
        seed_compare_plot_path,
    )
    seed_compare_csv_path = output_dir / "denoising_steps_seed_compare_summary.csv"
    _save_denoising_seed_compare_summary_csv(
        real_cpu,
        generated_by_condition,
        seed_frames_by_condition,
        seed_compare_csv_path,
    )

    video_rows = [("GT", real_cpu)]
    for step_count in sorted(generated_by_step):
        video_rows.append((f"{step_count} steps", generated_by_step[step_count]))
    comparison_video = _build_labeled_rows_video(video_rows)
    comparison_path = output_dir / "denoising_steps_comparison.mp4"
    _save_frames_as_video(comparison_video, comparison_path, display_scale=2)

    error_video = _build_raw_error_rows_video(
        real_cpu,
        generated_by_step,
        args.compare_error_gain,
    )
    error_video_path = output_dir / "denoising_steps_raw_error_comparison.mp4"
    _save_frames_as_video(error_video, error_video_path, display_scale=2)

    print(
        "Denoising-step comparison complete: "
        f"sample_idx={sample_idx} seed_frames={rollout_seed_frames} "
        f"total_frames={args.rollout_total_frames} "
        f"steps={args.compare_denoising_steps} "
        f"plot={plot_path} csv={csv_path} "
        f"seed_compare_plot={seed_compare_plot_path} "
        f"seed_compare_csv={seed_compare_csv_path} "
        f"video={comparison_path} error_video={error_video_path}"
    )
    print("Denoising-step post-seed metrics:")
    for step_count in sorted(metrics_summary):
        metrics = metrics_summary[step_count]
        print(
            f"  steps={step_count}: "
            f"mean_mse={metrics['mean_mse']:.6f} "
            f"final_mse={metrics['final_mse']:.6f} "
            f"mean_psnr={metrics['mean_psnr']:.2f} "
            f"final_psnr={metrics['final_psnr']:.2f}"
        )


def _compare_rollout_strategies(
    model,
    test_dataloader: VideoWindowLoader,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    rollout_total_frames = 2 * model.num_images_in_video
    rollout_seed_frames = (
        args.rollout_seed_frames
        if args.rollout_seed_frames is not None
        else model.num_images_in_video
    )
    if not 1 <= rollout_seed_frames < rollout_total_frames:
        raise ValueError(
            "rollout_seed_frames must be at least 1 and less than "
            f"2 * num_images_in_video ({rollout_total_frames})"
        )
    if args.rollout_strategy_best_of_n < 1:
        raise ValueError("rollout_strategy_best_of_n must be at least 1")
    if args.rollout_strategy_keyframe_interval < 1:
        raise ValueError("rollout_strategy_keyframe_interval must be at least 1")
    if args.rollout_strategy_outlier_delta_multiplier <= 0:
        raise ValueError("rollout_strategy_outlier_delta_multiplier must be positive")

    batch = next(iter(test_dataloader))
    if batch.dim() != 5:
        raise ValueError("Expected batch shape (B, T, C, H, W)")
    if batch.shape[1] < rollout_total_frames:
        raise ValueError(
            f"Loader returned {batch.shape[1]} frames; need at least "
            f"{rollout_total_frames}"
        )

    if args.compare_sample_index is None:
        sample_idx = random.randint(0, batch.size(0) - 1)
    else:
        if not 0 <= args.compare_sample_index < batch.size(0):
            raise ValueError(
                f"compare_sample_index must be in [0, {batch.size(0) - 1}], "
                f"got {args.compare_sample_index}"
            )
        sample_idx = args.compare_sample_index

    real_video = batch[
        sample_idx : sample_idx + 1, :rollout_total_frames
    ].to(device)
    action_sequence = _get_action_sequence(model, real_video)
    rollout_actions = action_sequence[
        :, rollout_seed_frames - 1 : rollout_total_frames - 1
    ]
    expected_actions = rollout_total_frames - rollout_seed_frames
    if rollout_actions.shape[1] != expected_actions:
        raise ValueError(
            f"Expected {expected_actions} rollout actions, got "
            f"{rollout_actions.shape[1]}"
        )

    seed_video = real_video[:, :rollout_seed_frames]
    rows: list[tuple[str, torch.Tensor]] = [("GT", real_video[0].detach().cpu())]
    base_seed = args.rollout_strategy_seed

    for strategy_idx, strategy in enumerate(ROLLOUT_STRATEGIES):
        if base_seed is None:
            strategy_rng = random.Random()
        else:
            strategy_rng = random.Random(base_seed + strategy_idx)
        with torch.no_grad():
            generated = rollout_with_strategy(
                model,
                seed_video,
                rollout_actions,
                args.max_steps,
                strategy.name,
                strategy_rng,
                best_of_n=args.rollout_strategy_best_of_n,
                keyframe_interval=args.rollout_strategy_keyframe_interval,
                outlier_delta_multiplier=args.rollout_strategy_outlier_delta_multiplier,
            )
        if generated.shape[1] != rollout_total_frames:
            raise ValueError(
                f"Strategy {strategy.name} returned {generated.shape[1]} frames, "
                f"expected {rollout_total_frames}"
            )
        generated_cpu = generated[0].detach().cpu()
        rows.append((strategy.label, generated_cpu))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_video = _build_labeled_rows_video(rows)
    comparison_path = output_dir / "rollout_strategies_comparison.mp4"
    _save_frames_as_video(comparison_video, comparison_path, display_scale=2)

    print(
        "Rollout-strategy comparison complete: "
        f"sample_idx={sample_idx} seed_frames={rollout_seed_frames} "
        f"total_frames={rollout_total_frames} max_steps={args.max_steps} "
        f"actions={rollout_actions[0].detach().cpu().tolist()} "
        f"video={comparison_path}"
    )
    for strategy in ROLLOUT_STRATEGIES:
        print(f"  row={strategy.label} strategy={strategy.name}")


def _interactive_loop(
    model,
    test_dataloader: VideoWindowLoader,
    config: DynamicsModelTrainingConfig,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    logger.info("Interactive inference ready.")
    logger.info("Commands: <action_id> (int) to generate, 'new' for new video, 'q' to quit.")

    loader_iter = iter(test_dataloader)
    required_context = config.num_images_in_video - 1

    def _sample_new_video() -> tuple[torch.Tensor, Optional[int]]:
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_dataloader)
            batch = next(loader_iter)

        if batch.dim() != 5:
            raise ValueError("Expected batch shape (B, T, C, H, W)")
        idx = random.randint(0, batch.size(0) - 1)
        video = batch[idx : idx + 1].to(device)
        logger.info("Selected random video idx=%d from batch size=%d", idx, batch.size(0))
        actual_next_action = _get_actual_next_action(model, video, required_context)
        return video[:, :required_context, :, :, :], actual_next_action

    # Current sliding window of frames: (1, T, C, H, W)
    current_video: Optional[torch.Tensor] = None
    step_counter = 0
    session_id = 0
    output_dir = Path(args.output_dir)
    total_video: Optional[torch.Tensor] = None
    actual_next_action: Optional[int] = None

    while True:
        # If no current video, sample one
        if current_video is None:
            try:
                current_video, actual_next_action = _sample_new_video()
                total_video = current_video.clone()
                session_id += 1
                step_counter = 0
                prefix = f"session_{session_id:03d}"
                archive_path, session_live_path, live_path = _save_concatenated_frames(
                    current_video, output_dir, prefix, step_counter
                )
                action_summary = (
                    f" actual next action={actual_next_action}"
                    if actual_next_action is not None
                    else ""
                )
                print(
                    f"[session {session_id}] Initial frames saved to: {archive_path}"
                    f" | session live view: {session_live_path} | live view: {live_path}"
                    f"{action_summary}"
                )
            except Exception as exc:
                logger.error("Failed to sample initial video: %s", exc)
                continue

        prompt = "Action id (int)"
        if actual_next_action is not None:
            prompt += f" [actual next action: {actual_next_action}]"
        prompt += ", 'new', or 'q': "
        cmd = input(prompt).strip()
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if cmd.lower() == "new":
            current_video = None
            actual_next_action = None
            print("Switching to new video...")
            continue

        try:
            action_id = int(cmd)
        except ValueError:
            print("Please provide a valid integer for action id, 'new', or 'q'.")
            continue

        with torch.no_grad():
            try:
                action_tensor = torch.tensor(
                    [action_id], dtype=torch.long, device=device
                )
                result = model.predict_next_frame(
                    current_video, action_tensor, max_steps=args.max_steps
                )
            except NotImplementedError:
                logger.warning(
                    "DynamicsModel.predict_next_frame is not implemented."
                )
                continue
            except Exception as exc:
                traceback.print_exc()
                logger.error("Inference call failed: %s", exc)
                continue

        # result should be shape (1, 1, C, H, W) or (1, C, H, W) for the new frame
        # Update sliding window: drop oldest frame, append new frame
        if result.dim() == 4:
            new_frame = result.unsqueeze(1)  # (1, 1, C, H, W)
        elif result.dim() == 5:
            new_frame = result[:, -1:, :, :, :]  # take last generated frame
        else:
            new_frame = result.unsqueeze(0).unsqueeze(0)

        # Slide window: keep last (num_frames - 1) frames, append new frame
        current_video = torch.cat([current_video[:, 1:, :, :, :], new_frame], dim=1)

        if total_video is None:
            raise ValueError("total_video is None")

        total_video = torch.cat([total_video, new_frame], dim=1)

        step_counter += 1
        prefix = f"session_{session_id:03d}"
        archive_path, session_live_path, live_path = _save_concatenated_frames(
            total_video, output_dir, prefix, step_counter
        )
        actual_action_summary = (
            str(actual_next_action) if actual_next_action is not None else "n/a"
        )
        print(
            f"[session {session_id}, step {step_counter}] chosen_action={action_id} "
            f"actual_next_action={actual_action_summary} saved to: {archive_path} "
            f"| session live view: {session_live_path} | live view: {live_path}"
        )
        actual_next_action = None


def main():
    args = tyro.cli(InteractiveInferenceArgs)
    device = torch.device(args.device)
    config = _build_config(args)
    model = _load_dynamics_model(config, device)
    needs_rollout_window = args.mode in (
        "actual_actions_rollout",
        "spam_actions_grid",
        "compare_denoising_steps",
        "compare_rollout_strategies",
        "visualize_denoising_trace",
    )
    dataloader_frames = (
        2 * model.num_images_in_video
        if args.mode == "compare_rollout_strategies"
        else args.rollout_total_frames
        if needs_rollout_window
        else config.num_images_in_video
    )
    dataset_limit = args.rollout_dataset_limit if needs_rollout_window else None
    test_dataloader = _build_test_dataloader(
        config,
        dataloader_frames,
        dataset_limit,
        args.num_workers,
        args.rollout_split if needs_rollout_window else "eval",
    )
    if args.mode == "actual_actions_rollout":
        _actual_actions_rollout(model, test_dataloader, args, device)
    elif args.mode == "spam_actions_grid":
        _spam_actions_grid(model, test_dataloader, args, device)
    elif args.mode == "compare_denoising_steps":
        _compare_denoising_steps(model, test_dataloader, args, device)
    elif args.mode == "compare_rollout_strategies":
        _compare_rollout_strategies(model, test_dataloader, args, device)
    elif args.mode == "visualize_denoising_trace":
        _visualize_denoising_trace(model, test_dataloader, args, device)
    else:
        _interactive_loop(model, test_dataloader, config, args, device)


if __name__ == "__main__":
    main()

