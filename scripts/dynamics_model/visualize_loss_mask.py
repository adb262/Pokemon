import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from data.datasets.atari_pong.atari_pong_dataset import AtariPongDataset
from data.datasets.atari_pong.atari_pong_dataset_creator import AtariPongDatasetCreator
from data.datasets.cache import Cache
from data.datasets.data_types.open_world_types import OpenWorldVideoLog
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from dynamics_model.training_args import DynamicsModelTrainingConfig
from loss.loss_fns import (
    action_weight,
    build_weight_mask,
    compute_changed_pixel_mask,
    compute_target_residuals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_CONFIG = DynamicsModelTrainingConfig()


@dataclass
class LossMaskVisualizationArgs:
    dataset_type: Literal["pokemon", "atari_pong"] = BASE_CONFIG.dataset_type
    frames_dir: str = BASE_CONFIG.frames_dir
    atari_pong_data_dir: Optional[str] = BASE_CONFIG.atari_pong_data_dir
    atari_pong_crop_scoreboard: bool = BASE_CONFIG.atari_pong_crop_scoreboard
    atari_pong_require_full_gameplay: bool = BASE_CONFIG.atari_pong_require_full_gameplay
    image_size: int = BASE_CONFIG.image_size
    num_images_in_video: int = BASE_CONFIG.num_images_in_video
    frame_spacing: int = BASE_CONFIG.frame_spacing
    num_unique_frames: Optional[int] = BASE_CONFIG.num_unique_frames
    dataset_limit: int = BASE_CONFIG.dataset_limit
    dataset_train_key: Optional[str] = BASE_CONFIG.dataset_train_key
    use_s3: bool = BASE_CONFIG.use_s3
    sync_from_s3: bool = BASE_CONFIG.sync_from_s3
    local_cache_dir: Optional[str] = BASE_CONFIG.local_cache_dir
    max_cache_size: int = BASE_CONFIG.max_cache_size
    split: Literal["train", "test"] = "train"
    sample_idx: Optional[int] = None
    random_seed: Optional[int] = None
    max_transitions: Optional[int] = None
    output_path: str = "dynamics_model_results/loss_mask_visualization.png"


def _frame_to_rgb(frame: torch.Tensor) -> np.ndarray:
    return np.clip(frame.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def _signed_residual_to_rgb(residual: torch.Tensor) -> np.ndarray:
    residual_rgb = residual.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((residual_rgb + 1.0) / 2.0, 0.0, 1.0)


def _reconstruct_next_frame(
    previous_frame: torch.Tensor, residual: torch.Tensor
) -> np.ndarray:
    reconstructed = torch.clamp(previous_frame + residual, 0.0, 1.0)
    return _frame_to_rgb(reconstructed)


def _overlay_changed_pixels(frame: torch.Tensor, changed_mask: torch.Tensor) -> np.ndarray:
    frame_rgb = _frame_to_rgb(frame)
    mask = changed_mask.detach().cpu().numpy()[..., None]
    overlay = frame_rgb * (1.0 - 0.35 * mask)
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.65 * mask[..., 0], 0.0, 1.0)
    return overlay


def _weight_mask_to_rgb(weight_mask: torch.Tensor) -> np.ndarray:
    weight_np = weight_mask.detach().cpu().numpy()
    if float(action_weight) <= 1.0:
        normalized = np.clip(weight_np, 0.0, 1.0)
    else:
        denom = max(float(action_weight) - 1.0, 1.0)
        normalized = np.clip((weight_np - 1.0) / denom, 0.0, 1.0)

    low = np.array([35, 18, 55], dtype=np.float32)
    high = np.array([255, 210, 60], dtype=np.float32)
    rgb = low + (high - low) * normalized[..., None]
    return np.clip(rgb / 255.0, 0.0, 1.0)


def _rgb_array_to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _load_sample(dataset: object, sample_idx: int) -> tuple[torch.Tensor, int]:
    dataset_length = len(dataset)  # type: ignore[arg-type]
    for idx in range(sample_idx, dataset_length):
        sample = dataset[idx]  # type: ignore[index]
        if sample is not None:
            return sample, idx
    raise ValueError(
        f"Could not load a valid sample starting from index {sample_idx}. "
        f"Dataset length: {dataset_length}"
    )


def _choose_sample_idx(dataset: object, args: LossMaskVisualizationArgs) -> int:
    dataset_length = len(dataset)  # type: ignore[arg-type]
    if dataset_length == 0:
        raise ValueError("Dataset is empty")
    if args.sample_idx is not None:
        if not 0 <= args.sample_idx < dataset_length:
            raise ValueError(
                f"sample_idx must be in [0, {dataset_length - 1}], got {args.sample_idx}"
            )
        return args.sample_idx

    rng = random.Random(args.random_seed)
    return rng.randrange(dataset_length)


def _resolve_pokemon_dataset_key(
    dataset_train_key: Optional[str], split: Literal["train", "test"]
) -> str:
    if dataset_train_key is None:
        raise ValueError(
            "dataset_train_key is required for pokemon visualizations so the script "
            "can load a local dataset JSON without touching S3."
        )
    if split == "train":
        return dataset_train_key
    return dataset_train_key.replace("train", "test")


def _load_pokemon_dataset(
    args: LossMaskVisualizationArgs, local_cache: Cache
) -> OpenWorldRunningDataset:
    dataset_key = _resolve_pokemon_dataset_key(args.dataset_train_key, args.split)
    with open(dataset_key, "r") as f:
        dataset = OpenWorldVideoLog.model_validate_json(f.read())

    return OpenWorldRunningDataset(
        dataset=dataset,
        local_cache=local_cache,
        image_size=args.image_size,
        num_images_in_video=args.num_images_in_video,
        num_unique_frames=args.num_unique_frames,
        limit=args.dataset_limit if args.split == "train" else min(args.dataset_limit, 100),
    )


def _load_atari_dataset(args: LossMaskVisualizationArgs) -> AtariPongDataset:
    if args.atari_pong_data_dir is None:
        raise ValueError("atari_pong_data_dir is required when dataset_type=atari_pong")

    creator = AtariPongDatasetCreator(
        data_dir=args.atari_pong_data_dir,
        num_frames_in_video=args.num_images_in_video,
        limit=args.dataset_limit,
        image_size=args.image_size,
        frame_spacing=args.frame_spacing,
        require_full_gameplay=args.atari_pong_require_full_gameplay,
    )
    train_log, test_log = creator.setup()
    dataset = train_log if args.split == "train" else test_log
    return AtariPongDataset(
        dataset=dataset,
        image_size=args.image_size,
        num_images_in_video=args.num_images_in_video,
        crop_scoreboard=args.atari_pong_crop_scoreboard,
        limit=args.dataset_limit if args.split == "train" else min(args.dataset_limit, 100),
    )


def save_loss_mask_visualization(
    video: torch.Tensor,
    output_path: Path,
    *,
    max_transitions: Optional[int] = None,
    sample_idx: int,
) -> None:
    batched_video = video.unsqueeze(0)
    target_residuals = compute_target_residuals(batched_video).squeeze(0)
    changed_pixels = compute_changed_pixel_mask(target_residuals, 0.005).squeeze(0)
    weight_mask = build_weight_mask(
        changed_pixels,
        changed_weight=action_weight,
        unchanged_weight=0.0,
    )

    per_pixel_change_mask = changed_pixels.max(dim=1).values
    per_pixel_weight_mask = weight_mask.max(dim=1).values

    num_transitions = target_residuals.shape[0]
    if max_transitions is not None:
        num_transitions = min(num_transitions, max_transitions)

    row_labels = [
        "Previous frame",
        "Next frame",
        "Target residual\n(signed RGB)",
        "Prev + residual\n(reconstructed next)",
        f"Loss weight\n(1 or {action_weight})",
        "Weighted pixels\n(red overlay)",
    ]
    font = ImageFont.load_default()
    cell_width = video.shape[-1]
    cell_height = video.shape[-2]
    padding = 12
    row_label_width = 150
    title_height = 48
    subtitle_height = 34
    col_title_height = 34

    canvas_width = (
        row_label_width
        + padding
        + num_transitions * (cell_width + padding)
        + padding
    )
    canvas_height = (
        title_height
        + subtitle_height
        + col_title_height
        + len(row_labels) * (cell_height + padding)
        + padding
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (padding, padding),
        "Residual-target loss mask visualization",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (padding, padding + 18),
        "Residuals are frame[t+1] - frame[t]; reconstructed next = clamp(frame[t] + residual); red overlay marks pixels upweighted in the residual loss.",
        fill=(60, 60, 60),
        font=font,
    )

    grid_top = title_height + subtitle_height + col_title_height
    for row, label in enumerate(row_labels):
        row_top = grid_top + row * (cell_height + padding)
        draw.multiline_text(
            (padding, row_top + 4),
            label,
            fill=(0, 0, 0),
            font=font,
            spacing=2,
        )

    for col in range(num_transitions):
        weighted_percent = 100.0 * per_pixel_change_mask[col].float().mean().item()
        col_left = row_label_width + padding + col * (cell_width + padding)
        draw.multiline_text(
            (col_left, title_height + subtitle_height),
            f"t={col}->{col + 1}\nweighted {weighted_percent:.1f}%",
            fill=(0, 0, 0),
            font=font,
            spacing=2,
        )

        cell_images = [
            _rgb_array_to_image(_frame_to_rgb(video[col])),
            _rgb_array_to_image(_frame_to_rgb(video[col + 1])),
            _rgb_array_to_image(_signed_residual_to_rgb(target_residuals[col])),
            _rgb_array_to_image(
                _reconstruct_next_frame(video[col], target_residuals[col])
            ),
            _rgb_array_to_image(_weight_mask_to_rgb(per_pixel_weight_mask[col])),
            _rgb_array_to_image(
                _overlay_changed_pixels(video[col + 1], per_pixel_change_mask[col])
            ),
        ]
        for row, image in enumerate(cell_images):
            row_top = grid_top + row * (cell_height + padding)
            canvas.paste(image, (col_left, row_top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    logger.info("Saved visualization for sample %s to %s", sample_idx, output_path)


def main(args: LossMaskVisualizationArgs) -> None:
    if args.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")

    local_cache = Cache(max_size=args.max_cache_size, cache_dir=args.local_cache_dir)
    if args.dataset_type == "pokemon":
        dataset = _load_pokemon_dataset(args, local_cache)
    else:
        dataset = _load_atari_dataset(args)
    selected_sample_idx = _choose_sample_idx(dataset, args)
    video, loaded_sample_idx = _load_sample(dataset, selected_sample_idx)

    save_loss_mask_visualization(
        video,
        Path(args.output_path),
        max_transitions=args.max_transitions,
        sample_idx=loaded_sample_idx,
    )


if __name__ == "__main__":
    main(tyro.cli(LossMaskVisualizationArgs))
