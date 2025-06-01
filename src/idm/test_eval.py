#!/usr/bin/env python3
"""
Test Pokemon VQVAE Evaluation Script
Tests the evaluation functionality with a trained or randomly initialized model.
"""

from data_collection.pokemon_frame_loader import PokemonFrameLoader
from train import TrainingConfig
from vqvae import VQVAE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
from pathlib import Path
from PIL import Image

# Add the parent directory to the path so we can import from data_collection
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(config: TrainingConfig):
    """Create and initialize the VQVAE model"""
    num_patches = (config.image_size // config.patch_size) ** 2

    model = VQVAE(
        channels=3,
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        patch_embed_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        num_heads=config.num_heads,
        num_patches=num_patches,
        num_transformer_layers=config.num_transformer_layers,
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint with comprehensive error handling"""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.error("No model_state_dict found in checkpoint")
        return None

    # Extract metadata with fallbacks
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    loss = checkpoint.get('loss', 0.0)
    config = checkpoint.get('config', {})
    timestamp = checkpoint.get('timestamp', 'unknown')

    logger.info(f"Checkpoint loaded successfully:")
    logger.info(f"  Epoch: {epoch}, Batch: {batch_idx}")
    logger.info(f"  Loss: {loss:.6f}")
    logger.info(f"  Timestamp: {timestamp}")

    return {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'loss': loss,
        'config': config
    }


def tensor_to_image(tensor):
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


def test_visualization(
        model: VQVAE, dataloader: PokemonFrameLoader, device: torch.device, num_samples: int = 2,
        save_dir: str = "test_eval_results"):
    """Test visualization with model"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of data
        image1_batch, image2_batch = next(iter(dataloader))
        image1_batch = image1_batch[:num_samples].to(device)
        image2_batch = image2_batch[:num_samples].to(device)

        logger.info(f"Input shapes: {image1_batch.shape}, {image2_batch.shape}")

        # Forward pass
        try:
            residual, indices = model.inference_step(image1_batch, image2_batch)
            reconstructed, indices = image2_batch + residual
            logger.info(f"Reconstruction shape: {reconstructed.shape}")

            # Create visualization
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                # Input frame 1
                img1 = tensor_to_image(image1_batch[i])
                axes[i, 0].imshow(img1)
                axes[i, 0].set_title(f"Input Frame 1 (Sample {i+1})")
                axes[i, 0].axis('off')

                # Input frame 2 (target)
                img2 = tensor_to_image(image2_batch[i])
                axes[i, 1].imshow(img2)
                axes[i, 1].set_title(f"Target Frame 2 (Sample {i+1})")
                axes[i, 1].axis('off')

                # Reconstructed frame 2
                recon = tensor_to_image(reconstructed[i])
                axes[i, 2].imshow(recon)
                axes[i, 2].set_title(f"Reconstructed Frame 2 (Sample {i+1}). Action: {indices[i]}")
                axes[i, 2].axis('off')

                # Save individual images
                img1.save(os.path.join(save_dir, f"sample_{i+1}_input1.png"))
                img2.save(os.path.join(save_dir, f"sample_{i+1}_target.png"))
                recon.save(os.path.join(save_dir, f"sample_{i+1}_reconstructed.png"))

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "test_reconstruction_comparison.png"), dpi=150, bbox_inches='tight')
            plt.show()

            # Calculate and display reconstruction loss
            mse_loss = nn.MSELoss()(reconstructed, image2_batch)
            logger.info(f"Reconstruction MSE Loss: {mse_loss.item():.6f}")

            return mse_loss.item()

        except Exception as e:
            logger.error(f"Error during reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Pokemon VQVAE Evaluation')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint to load')
    parser.add_argument('--frames-dir', default='pokemon_frames', help='Path to frames directory')
    parser.add_argument('--image-size', type=int, default=400, help='Image size')
    parser.add_argument('--num-samples', type=int, default=2, help='Number of samples to visualize')
    parser.add_argument('--save-dir', default='test_eval_results', help='Directory to save results')

    args = parser.parse_args()

    # Configuration
    config = TrainingConfig()
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Create data loader
    logger.info("Creating data loader...")
    dataloader = PokemonFrameLoader(
        frames_dir=args.frames_dir,
        batch_size=args.num_samples,
        image_size=args.image_size,
        shuffle=True,
        num_workers=2,
        min_frame_gap=1,
        max_frame_gap=3,
        stage="test"
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logger.info("Dataset Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Try to load checkpoint
    checkpoint_loaded = False
    checkpoint_info = None

    # Check for specific checkpoint first
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading specified checkpoint: {args.checkpoint}")
        checkpoint_info = load_checkpoint(model, args.checkpoint, device)
        if checkpoint_info:
            checkpoint_loaded = True

    # Try common checkpoint locations if no specific checkpoint provided
    if not checkpoint_loaded:
        checkpoint_paths = [
            "checkpoints/checkpoint_latest.pt",
            "checkpoints/checkpoint_best.pt"
        ]

        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                logger.info(f"Found checkpoint: {checkpoint_path}")
                checkpoint_info = load_checkpoint(model, checkpoint_path, device)
                if checkpoint_info:
                    checkpoint_loaded = True
                    break

    model.to(device)

    if checkpoint_loaded and checkpoint_info:
        logger.info("=" * 50)
        logger.info("CHECKPOINT INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Epoch: {checkpoint_info['epoch']}")
        logger.info(f"Batch: {checkpoint_info['batch_idx']}")
        logger.info(f"Training Loss: {checkpoint_info['loss']:.6f}")

        if 'config' in checkpoint_info and checkpoint_info['config']:
            config_info = checkpoint_info['config'].__dict__
            logger.info(f"Experiment: {config_info.get('experiment_name', 'unknown')}")
            logger.info(f"Training Image Size: {config_info.get('image_size', 'unknown')}")
            logger.info(f"Training Batch Size: {config_info.get('batch_size', 'unknown')}")
    else:
        logger.info("No checkpoint loaded - using randomly initialized model")
        logger.info("Note: Reconstruction quality will be poor with untrained model")

    logger.info("=" * 50)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 50)

    # Test visualization
    logger.info("Generating test visualizations...")
    recon_loss = test_visualization(
        model, dataloader, device,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )

    if recon_loss is not None:
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Reconstruction Loss: {recon_loss:.6f}")
        logger.info(f"Results saved to: {args.save_dir}")

        if checkpoint_loaded and checkpoint_info:
            train_loss = checkpoint_info['loss']
            if recon_loss < train_loss:
                logger.info(f"✓ Model performing better than training: {train_loss - recon_loss:.6f} improvement")
            else:
                logger.info(f"⚠ Model performing worse than training: {recon_loss - train_loss:.6f} degradation")

        logger.info("Check the saved images to visually assess reconstruction quality")
    else:
        logger.error("✗ Evaluation failed")


if __name__ == "__main__":
    main()
