"""Render a post-train-tokenizer rollout as an MP4 triptych.

This script intentionally uses ``post_train_tokenizer.rollout_with_trainable_decoder``
for generation, then encodes the resulting GT | Pred | Abs Err frames as MP4.
"""
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from monitoring.videos import convert_video_to_images
from scripts.video_tokenizer.post_train_tokenizer import (
    PostTrainTokenizerConfig,
    _apply_dynamics_checkpoint_config,
    _load_checkpoint_config,
    _make_frame_triptych,
    load_frozen_dynamics_stack,
    rollout_with_trainable_decoder,
)
from video_tokenization.checkpoints import load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class PostTrainRolloutMp4Config:
    tokenizer_checkpoint_path: str
    dynamics_model_checkpoint_path: str
    action_model_checkpoint_path: Optional[str] = None

    output_path: str = "post_trained/post_train_rollout_mp4/post_train_rollout_triptych.mp4"
    post_train_frames: int = 32
    rollout_seed_frames: int = 1
    dynamics_context_frames: int = 16
    max_denoising_steps: int = 1
    batch_size: int = 1
    num_videos: int = 1
    seed: int = 42
    test_dataset_limit: Optional[int] = 8
    num_workers: int = 0

    local_cache_dir: Optional[str] = "cache"
    max_cache_size: int = 100000
    dataset_limit: int = 10
    atari_pong_require_full_gameplay: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16: bool = True
    use_compile: bool = False
    reconstruction_error_scale: float = 5.0
    fps: int = 6
    display_scale: int = 4


def _build_post_train_config(
    config: PostTrainRolloutMp4Config,
) -> PostTrainTokenizerConfig:
    post_train_config = PostTrainTokenizerConfig(
        tokenizer_checkpoint_path=config.tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=config.dynamics_model_checkpoint_path,
        action_model_checkpoint_path=config.action_model_checkpoint_path,
        post_train_frames=config.post_train_frames,
        dynamics_context_frames=config.dynamics_context_frames,
        rollout_seed_frames=config.rollout_seed_frames,
        max_denoising_steps=config.max_denoising_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        test_dataset_limit=config.test_dataset_limit,
        local_cache_dir=config.local_cache_dir,
        max_cache_size=config.max_cache_size,
        dataset_limit=config.dataset_limit,
        atari_pong_require_full_gameplay=config.atari_pong_require_full_gameplay,
        device=config.device,
        use_bf16=config.use_bf16,
        use_compile=config.use_compile,
        reconstruction_error_scale=config.reconstruction_error_scale,
        logging_backend="none",
        use_wandb=False,
    )

    dynamics_config = _load_checkpoint_config(config.dynamics_model_checkpoint_path)
    action_config = _load_checkpoint_config(config.action_model_checkpoint_path)
    _apply_dynamics_checkpoint_config(
        post_train_config,
        dynamics_config,
        action_config,
    )
    post_train_config.dynamics_context_frames = min(
        post_train_config.dynamics_context_frames,
        post_train_config.num_images_in_video,
    )
    post_train_config.atari_pong_require_full_gameplay = (
        config.atari_pong_require_full_gameplay
    )
    return post_train_config


def _save_pil_frames_as_mp4(
    frames,
    output_path: Path,
    fps: int,
    display_scale: int,
) -> None:
    if not frames:
        raise ValueError("Cannot save an empty MP4")
    if display_scale < 1:
        raise ValueError(f"display_scale must be at least 1, got {display_scale}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    raw_video = b"".join(
        np.asarray(frame.convert("RGB")).tobytes()
        for frame in frames
    )
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
        f"{width}x{height}",
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
        str(output_path),
    ]
    try:
        subprocess.run(
            ffmpeg_cmd,
            input=raw_video,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to save browser-compatible MP4s") from exc
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to save video to {output_path}: {error_message}"
        ) from exc


def _output_path_for_index(output_path: Path, video_idx: int, num_videos: int) -> Path:
    if num_videos == 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}_{video_idx:03d}{output_path.suffix}")


def _make_triptych_frames_for_sample(
    full_rollout: torch.Tensor,
    gt_full: torch.Tensor,
    reconstruction_error_scale: float,
    sample_idx: int,
):
    sample_rollout = full_rollout[sample_idx : sample_idx + 1]
    sample_gt = gt_full[sample_idx : sample_idx + 1]
    sample_error = sample_rollout - sample_gt

    pred_images = convert_video_to_images(sample_rollout)
    gt_images = convert_video_to_images(sample_gt)
    error_images = convert_video_to_images(
        sample_error,
        value_mode="magnitude",
        residual_scale=reconstruction_error_scale,
    )
    return [
        _make_frame_triptych(
            gt_images[0][frame_idx],
            pred_images[0][frame_idx],
            error_images[0][frame_idx],
        )
        for frame_idx in range(len(gt_images[0]))
    ]


def main(config: PostTrainRolloutMp4Config) -> None:
    post_train_config = _build_post_train_config(config)
    if post_train_config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")
    if config.num_videos < 1:
        raise ValueError(f"num_videos must be at least 1, got {config.num_videos}")
    if post_train_config.post_train_frames <= post_train_config.rollout_seed_frames:
        raise ValueError(
            "post_train_frames must be greater than rollout_seed_frames: "
            f"{post_train_config.post_train_frames} <= "
            f"{post_train_config.rollout_seed_frames}"
        )

    device = torch.device(post_train_config.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(post_train_config.seed)
    torch.cuda.manual_seed_all(post_train_config.seed)

    logger.info(
        "Loading tokenizer from %s",
        post_train_config.tokenizer_checkpoint_path,
    )
    trainable_tokenizer, tokenizer_config = load_model_from_checkpoint(
        post_train_config.tokenizer_checkpoint_path,
        device,
    )
    post_train_config.image_size = tokenizer_config.image_size
    post_train_config.patch_size = tokenizer_config.patch_size
    post_train_config.bins = list(tokenizer_config.bins)
    trainable_tokenizer.eval()

    logger.info("Loading frozen dynamics stack")
    dynamics_model = load_frozen_dynamics_stack(post_train_config, device)

    local_cache = Cache(
        max_size=post_train_config.max_cache_size,
        cache_dir=post_train_config.local_cache_dir,
    )
    _, test_dataset = build_datasets(
        post_train_config,
        local_cache,
        num_frames_in_video=post_train_config.post_train_frames,
        test_limit=post_train_config.test_dataset_limit,
    )
    test_dataloader = VideoWindowLoader(
        dataset=test_dataset,
        batch_size=post_train_config.batch_size,
        image_size=post_train_config.image_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=post_train_config.seed,
    )

    use_amp = post_train_config.use_bf16 and device.type == "cuda"
    output_path = Path(config.output_path)
    saved_count = 0

    for video_batch in test_dataloader:
        if saved_count >= config.num_videos:
            break

        video_batch = video_batch.to(device, non_blocking=True)
        video_batch = video_batch[:, : post_train_config.post_train_frames]

        with torch.no_grad():
            rollout_decoded, _, _ = rollout_with_trainable_decoder(
                dynamics_model,
                trainable_tokenizer,
                video_batch,
                post_train_config,
                use_amp,
            )

        full_rollout = torch.cat(
            [
                video_batch[:, : post_train_config.rollout_seed_frames],
                rollout_decoded,
            ],
            dim=1,
        )
        gt_full = video_batch[:, : post_train_config.post_train_frames]

        for sample_idx in range(video_batch.shape[0]):
            if saved_count >= config.num_videos:
                break

            triptych_frames = _make_triptych_frames_for_sample(
                full_rollout,
                gt_full,
                post_train_config.reconstruction_error_scale,
                sample_idx,
            )
            sample_output_path = _output_path_for_index(
                output_path,
                saved_count,
                config.num_videos,
            )
            _save_pil_frames_as_mp4(
                triptych_frames,
                sample_output_path,
                config.fps,
                config.display_scale,
            )
            saved_count += 1
            logger.info(
                "Saved post-train rollout MP4 %d/%d to %s",
                saved_count,
                config.num_videos,
                sample_output_path,
            )

    if saved_count < config.num_videos:
        raise ValueError(
            f"Requested {config.num_videos} videos, but only saved {saved_count}. "
            "Increase the dataset limits or reduce num_videos."
        )


if __name__ == "__main__":
    main(tyro.cli(PostTrainRolloutMp4Config))
