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
import torch.nn as nn
from PIL import Image

from data.data_loaders.pokemon_frame_loader import PokemonFrameLoader
from idm.multi_frame_vqvae import MultiFrameVQVAE
from idm.training_args import TrainingConfig
from s3.s3_utils import get_s3_manager_from_env, parse_s3_path

TrainingConfig = TrainingConfig()

# Add the parent directory to the path so we can import from data_collection
sys.path.append(str(Path(__file__).parent.parent))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(config):
    """Create and initialize the VQVAE model"""
    num_patches = (config["image_size"] // config["patch_size"]) ** 2

    model = MultiFrameVQVAE(
        channels=3,
        image_size=(config["image_size"], config["image_size"]),
        patch_size=(config["patch_size"], config["patch_size"]),
        patch_embed_dim=128,
        hidden_dim=256,
        latent_dim=128,
        num_embeddings=8,
        num_heads=4,
        num_patches=num_patches,
        num_transformer_layers=2,
    )

    return model


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

    config = checkpoint["config"]
    model = create_model(config)
    model.to(device)

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
        logger.info(f"  Training config: {config.get('experiment_name', 'unknown')}")

    return {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss": loss,
        "config": config,
        "timestamp": timestamp,
    }


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

    return Image.fromarray(image_array)


def visualize_reconstruction(
    model, dataloader, device, num_samples=4, save_dir="eval_results"
):
    """Visualize reconstruction results"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of data
        image1_batch, image2_batch = next(iter(dataloader))
        image1_batch = image1_batch[:num_samples].to(device)
        image2_batch = image2_batch[:num_samples].to(device)

        # Forward pass
        try:
            residual, indices = model.inference_step(image1_batch, image2_batch)
            reconstructed = image2_batch + residual

            # Create visualization
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                # Input frame 1
                img1 = tensor_to_image(image1_batch[i])
                axes[i, 0].imshow(img1)
                axes[i, 0].set_title(f"Input Frame 1 (Sample {i + 1})")
                axes[i, 0].axis("off")

                # Input frame 2 (target)
                img2 = tensor_to_image(image2_batch[i])
                axes[i, 1].imshow(img2)
                axes[i, 1].set_title(f"Target Frame 2 (Sample {i + 1})")
                axes[i, 1].axis("off")

                # Reconstructed frame 2
                recon = tensor_to_image(reconstructed[i])
                axes[i, 2].imshow(recon)
                axes[i, 2].set_title(
                    f"Reconstructed Frame 2 (Sample {i + 1}). Action: {indices[i]}"
                )
                axes[i, 2].axis("off")

                # Save individual images
                img1.save(os.path.join(save_dir, f"sample_{i + 1}_input1.png"))
                img2.save(os.path.join(save_dir, f"sample_{i + 1}_target.png"))
                recon.save(os.path.join(save_dir, f"sample_{i + 1}_reconstructed.png"))

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "reconstruction_comparison.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()

            # Calculate and display reconstruction loss
            mse_loss = nn.MSELoss()(reconstructed, image2_batch)
            logger.info(f"Reconstruction MSE Loss: {mse_loss.item():.6f}")

            return mse_loss.item()

        except Exception as e:
            logger.error(f"Error during reconstruction: {e}")
            return None


def evaluate_model_metrics(
    model: MultiFrameVQVAE,
    dataloader: PokemonFrameLoader,
    device: torch.device,
    num_batches: int = 10,
):
    """Evaluate model on multiple batches and compute metrics"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (image1_batch, image2_batch) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            image1_batch = image1_batch.to(device)
            image2_batch = image2_batch.to(device)

            try:
                reconstructed, indices = model.inference_step(
                    image1_batch, image2_batch
                )
                loss = criterion(reconstructed, image2_batch)

                total_loss += loss.item() * image1_batch.size(0)
                total_samples += image1_batch.size(0)

                if batch_idx % 5 == 0:
                    logger.info(
                        f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}"
                    )

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    logger.info(
        f"Average reconstruction loss over {total_samples} samples: {avg_loss:.6f}"
    )

    return avg_loss


def main():
    """Main evaluation function"""
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

    # Configuration
    config = {
        "image_size": args.image_size,
        "patch_size": 16,
        "device": args.device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    }

    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")

    # Create data loader
    logger.info("Creating data loader...")
    dataloader = PokemonFrameLoader(
        frames_dir=args.frames_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        num_workers=2,
        min_frame_gap=1,
        max_frame_gap=3,
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Create and load model
    logger.info("Creating model...")
    model = create_model(config)
    model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    s3_manager = get_s3_manager_from_env()
    checkpoint_info = load_checkpoint(args.checkpoint, device, s3_manager)
    if checkpoint_info is None:
        logger.error("Failed to load checkpoint")
        return

    # Extract checkpoint information
    epoch = checkpoint_info["epoch"]
    batch_idx = checkpoint_info["batch_idx"]
    train_loss = checkpoint_info["loss"]
    checkpoint_config = checkpoint_info.get("config", {})

    # Check if checkpoint config matches current config
    if checkpoint_config:
        checkpoint_image_size = checkpoint_config.get("image_size", args.image_size)
        checkpoint_patch_size = checkpoint_config.get("patch_size", 16)

        if checkpoint_image_size != args.image_size:
            logger.warning(
                f"Image size mismatch: checkpoint={checkpoint_image_size}, current={args.image_size}"
            )
        if checkpoint_patch_size != config["patch_size"]:
            logger.warning(
                f"Patch size mismatch: checkpoint={checkpoint_patch_size}, current={config['patch_size']}"
            )

    logger.info(
        f"Model loaded from epoch {epoch}, batch {batch_idx} with training loss {train_loss:.6f}"
    )

    # Visualize reconstructions
    logger.info("Generating reconstruction visualizations...")
    recon_loss = visualize_reconstruction(
        model, dataloader, device, num_samples=args.num_samples, save_dir=args.save_dir
    )

    if recon_loss is not None:
        logger.info(f"Visualization complete. Results saved to {args.save_dir}")

    # Evaluate model metrics
    logger.info("Evaluating model metrics...")
    avg_loss = evaluate_model_metrics(model, dataloader, device, args.num_eval_batches)

    # Summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Training epoch: {epoch}")
    logger.info(f"Training batch: {batch_idx}")
    logger.info(f"Training loss: {train_loss:.6f}")
    logger.info(f"Evaluation loss: {avg_loss:.6f}")
    logger.info(
        f"Sample reconstruction loss: {recon_loss:.6f}"
        if recon_loss
        else "Sample reconstruction: Failed"
    )
    logger.info(f"Results saved to: {args.save_dir}")

    # Performance comparison
    if recon_loss is not None:
        improvement = train_loss - avg_loss
        if improvement > 0:
            logger.info(f"Model shows improvement: {improvement:.6f} loss reduction")
        else:
            logger.info(
                f"Model shows degradation: {abs(improvement):.6f} loss increase"
            )


if __name__ == "__main__":
    main()
