import argparse
import logging
import os
from typing import List, Tuple

import torch

from data.data_loaders.pokemon_open_world_loader import PokemonOpenWorldLoader
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)
from monitoring.frechet_distance import compute_frechet_distance
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.eval import convert_video_to_images
from video_tokenization.tokenizer import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_dataloaders(
    config: VideoTokenizerTrainingConfig, num_eval_samples: int
) -> Tuple[PokemonOpenWorldLoader, PokemonOpenWorldLoader]:
    """
    Build train and test dataloaders, mirroring the logic in train.py/eval.py,
    but optionally limiting the dataset size for faster comparison.
    """
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")

    logger.info("Creating cache and datasets...")
    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )

    dataset_creator = OpenWorldRunningDatasetCreator(
        dataset_dir=config.frames_dir,
        num_frames_in_video=config.num_images_in_video,
        output_log_json_file_name="log_dir_10000.json",
        local_cache=local_cache,
        limit=max(num_eval_samples, 1000),
        image_size=config.image_size,
    )

    train_dataset_raw, test_dataset_raw = dataset_creator.setup(train_percentage=0.9)

    train_dataset = OpenWorldRunningDataset(
        dataset=train_dataset_raw,
        local_cache=local_cache,
        image_size=config.image_size,
    )

    test_dataset = OpenWorldRunningDataset(
        dataset=test_dataset_raw,
        local_cache=local_cache,
        image_size=config.image_size,
    )

    logger.info("Creating dataloaders...")
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

    return train_dataloader, test_dataloader


def evaluate_model_on_subset(
    model: VideoTokenizer,
    dataloader: PokemonOpenWorldLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a model over a small subset of the dataset and collect real and
    reconstructed frames for Frechet distance computation.
    """
    real_batches: List[torch.Tensor] = []
    pred_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, video_batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            video_batch = video_batch.to(device)
            decoded: torch.Tensor = model(video_batch)

            if decoded.dim() == 5 and video_batch.dim() == 5:
                real_batches.append(video_batch.detach().cpu())
                pred_batches.append(decoded.detach().cpu())

    if not real_batches or not pred_batches:
        raise RuntimeError("No batches collected for Frechet computation.")

    real_all = torch.cat(real_batches, dim=0)
    pred_all = torch.cat(pred_batches, dim=0)
    return real_all, pred_all


def build_comparison_grid(
    models: List[VideoTokenizer],
    dataloader: PokemonOpenWorldLoader,
    model_names: List[str],
    device: torch.device,
    num_samples: int,
    output_dir: str,
) -> None:
    """
    Create an image grid comparing reconstructions from multiple models
    against the original input frames.

    The grid layout is:
        row 0: originals
        row 1: model 1 recon
        row 2: model 2 recon
        ...
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Get a single batch and slice to num_samples
    batch = next(iter(dataloader))
    batch = batch.to(device)
    batch = batch[:num_samples]

    with torch.no_grad():
        decoded_list = [model(batch).detach().cpu() for model in models]
        originals = batch.detach().cpu()

    original_videos = convert_video_to_images(originals)
    decoded_videos_list = [convert_video_to_images(decoded) for decoded in decoded_list]

    if not original_videos:
        raise RuntimeError("No videos available to build comparison grid.")

    # Clamp to actual available samples / frames to avoid index errors
    actual_num_samples = min(num_samples, len(original_videos))
    num_frames = min(len(original_videos[0]), *(len(v) for v in original_videos))
    num_rows = 1 + len(models)  # originals + one row per model

    fig, axs = plt.subplots(
        num_rows * actual_num_samples,
        num_frames,
        figsize=(num_frames * 2.5, num_rows * actual_num_samples * 2.5),
    )

    if isinstance(axs, np.ndarray) and axs.ndim == 1:
        axs = np.expand_dims(axs, 0)

    for sample_idx in range(actual_num_samples):
        # Row 0: originals
        for frame_idx in range(num_frames):
            ax = axs[sample_idx * num_rows + 0, frame_idx]
            ax.imshow(original_videos[sample_idx][frame_idx])
            if sample_idx == 0:
                ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

        # Subsequent rows: reconstructions from each model
        for model_idx, decoded_videos in enumerate(decoded_videos_list, start=1):
            for frame_idx in range(num_frames):
                ax = axs[sample_idx * num_rows + model_idx, frame_idx]
                ax.imshow(decoded_videos[sample_idx][frame_idx])
                if frame_idx == 0:
                    ax.set_ylabel(
                        model_names[model_idx - 1],
                        rotation=0,
                        labelpad=40,
                        fontsize=8,
                    )
                ax.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "models_comparison_grid.png")
    plt.savefig(grid_path)
    plt.close(fig)
    logger.info(f"Saved comparison grid to {grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple VideoTokenizer checkpoints on a small dataset subset."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths to compare.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        help="Optional human-readable labels for each checkpoint.",
    )
    parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=4,
        help="Number of batches to use for Frechet computation.",
    )
    parser.add_argument(
        "--num_grid_samples",
        type=int,
        default=4,
        help="Number of samples to visualize in the comparison grid.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_comparisons",
        help="Directory to save comparison outputs.",
    )
    # Optional overrides for dataset/model config
    parser.add_argument(
        "--frames_dir",
        type=str,
        help="Override frames directory used to build the dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size used for the dataloader.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Override image size used when loading frames.",
    )
    parser.add_argument(
        "--num_images_in_video",
        type=int,
        help="Override number of frames per video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override device string (e.g., 'cuda', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--local_cache_dir",
        type=str,
        help="Override local cache directory.",
    )
    parser.add_argument(
        "--max_cache_size",
        type=int,
        help="Override maximum cache size.",
    )

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_paths = args.checkpoints
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path must be provided.")

    # Load the first model and its config from checkpoint so that FSQ bins and
    # other hyperparameters match training.
    # The device may be overridden via CLI; otherwise use the checkpoint config's device.
    provisional_device = torch.device("cpu")
    first_model, first_config = load_model_from_checkpoint(
        checkpoint_paths[0], provisional_device
    )

    if args.device is not None:
        first_config.device = args.device

    # Apply dataset-related overrides on top of checkpoint config
    if args.frames_dir is not None:
        first_config.frames_dir = args.frames_dir
    if args.batch_size is not None:
        first_config.batch_size = args.batch_size
    if args.image_size is not None:
        first_config.image_size = args.image_size
    if args.num_images_in_video is not None:
        first_config.num_images_in_video = args.num_images_in_video
    if args.local_cache_dir is not None:
        first_config.local_cache_dir = args.local_cache_dir
    if args.max_cache_size is not None:
        first_config.max_cache_size = args.max_cache_size

    device = torch.device(first_config.device)
    first_model.to(device)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Using device: {device}")

    # Build dataloaders using the (possibly overridden) first checkpoint config
    _, test_dataloader = build_dataloaders(first_config, num_eval_samples=1000)

    # Build model labels
    if args.labels and len(args.labels) == len(checkpoint_paths):
        model_labels = args.labels
    else:
        model_labels = [os.path.basename(p) for p in checkpoint_paths]

    # Load models. We already loaded the first one to get its config.
    models: List[VideoTokenizer] = [first_model]
    for ckpt in checkpoint_paths[1:]:
        model, _ = load_model_from_checkpoint(ckpt, device)
        models.append(model)

    # Frechet distance per model
    frechet_results = {}
    for ckpt, label, model in zip(checkpoint_paths, model_labels, models):
        logger.info(f"Evaluating model {label} ({ckpt})...")
        real_all, pred_all = evaluate_model_on_subset(
            model, test_dataloader, device, max_batches=args.num_eval_batches
        )
        fd = compute_frechet_distance(real_all, pred_all)
        frechet_results[label] = fd
        logger.info(f"Frechet distance for {label}: {fd:.4f}")

    # Log summary
    logger.info("Frechet distance summary:")
    for label, fd in frechet_results.items():
        logger.info(f"  {label}: {fd:.4f}")

    # Build comparison grid
    build_comparison_grid(
        models=models,
        dataloader=test_dataloader,
        model_names=model_labels,
        device=device,
        num_samples=args.num_grid_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
