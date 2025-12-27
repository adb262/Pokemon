#!/usr/bin/env python3
"""
Pokemon VQVAE Evaluation Script
Loads a trained model and visualizes reconstruction results.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data.data_loaders.pokemon_frame_loader import PokemonFrameLoader
from data.s3.s3_utils import get_s3_manager_from_env, parse_s3_path
from latent_action_model.laqa import LatentActionQuantization
from latent_action_model.train_laqa import create_model
from latent_action_model.training_args import VideoTrainingConfig

# Add the parent directory to the path so we can import from data_collection
sys.path.append(str(Path(__file__).parent.parent))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device, s3_manager):
    """Load model checkpoint with support for new comprehensive format"""
    tmp_path = Path(f"{checkpoint_path.split('/')[-1]}")
    if checkpoint_path.startswith("s3://"):
        if not os.path.exists(tmp_path):
            bucket, key = parse_s3_path(checkpoint_path)
            s3_manager.download_file(key, tmp_path)

        checkpoint_path = tmp_path

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    config = VideoTrainingConfig(**checkpoint["config"])
    model = create_model(config)
    model.to(device).eval()

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.error("No model_state_dict found in checkpoint")
        return None

    # Extract metadata with fallbacks for older checkpoint formats
    epoch = checkpoint.get("epoch", 0)
    batch_idx = checkpoint.get("batch_idx", 0)
    loss = checkpoint.get("loss", 0.0)
    timestamp = checkpoint.get("timestamp", "unknown")

    logger.info("Checkpoint loaded successfully:")
    logger.info(f"  Path: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Batch: {batch_idx}")
    logger.info(f"  Loss: {loss:.6f}")
    logger.info(f"  Timestamp: {timestamp}")

    if config:
        logger.info(f"  Training config: {config.experiment_name}")

    return model, config


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image"""
    # tensor shape: (C, H, W) with values in [0, 1]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    # Clamp values to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    image_array = tensor.permute(1, 2, 0).cpu().numpy()

    # Convert to uint8
    image_array = (image_array * 255).astype(np.uint8)

    return Image.fromarray(image_array).convert("RGB")


def visualize_reconstruction(
    model: LatentActionQuantization,
    dataloader: PokemonFrameLoader,
    num_images_in_video: int,
    device: torch.device,
    num_samples: int = 4,
    save_dir: str = "eval_results",
):
    """Visualize reconstruction results"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of data
        video_batch = next(iter(dataloader))
        video_batch = video_batch.to(device)

        # Forward pass
        residual = model.inference(video_batch.permute(0, 2, 1, 3, 4))

        # Create visualization
        fig, axes = plt.subplots(
            num_samples, num_images_in_video + 1, figsize=(15, 5 * num_samples)
        )
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Input frames
            for j in range(num_images_in_video):
                img = tensor_to_image(video_batch[i, j])
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Input Frame {j + 1} (Sample {i + 1})")
                axes[i, j].axis("off")

            # Reconstructed frame
            recon = tensor_to_image(residual[i])
            axes[i, -1].imshow(recon)
            axes[i, -1].set_title(f"Reconstructed Frame (Sample {i + 1})")
            axes[i, -1].axis("off")

            # Save individual images
            for j in range(num_images_in_video):
                img = tensor_to_image(video_batch[i, j])
                img.save(os.path.join(save_dir, f"sample_{i + 1}_input{j + 1}.png"))
            recon.save(os.path.join(save_dir, f"sample_{i + 1}_reconstructed.png"))

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "reconstruction_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pokemon VQVAE Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--frames-dir", default="pokemon_frames", help="Path to frames directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument("--image-size", type=int, default=400, help="Image size")
    parser.add_argument(
        "--num-samples", type=int, default=4, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=10,
        help="Number of batches for metric evaluation",
    )
    parser.add_argument(
        "--save-dir", default="eval_results", help="Directory to save results"
    )
    parser.add_argument(
        "--device", default=None, help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    s3_manager = get_s3_manager_from_env()
    model, config = load_checkpoint(args.checkpoint, device, s3_manager)
    if model is None:
        logger.error("Failed to load checkpoint")
        return

    logger.info(f"Using device: {device}")

    # Create data loader
    logger.info("Creating data loader...")
    dataloader = PokemonFrameLoader(
        num_frames_in_video=config.num_images_in_video,
        frames_dir=args.frames_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        num_workers=2,
        limit=100,
        stage="train",
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Visualize reconstructions
    logger.info("Generating reconstruction visualizations...")
    visualize_reconstruction(
        model,
        dataloader,
        config.num_images_in_video,
        device,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
