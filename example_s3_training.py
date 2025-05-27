#!/usr/bin/env python3
"""
Example script for running Pokemon VQVAE training with S3 storage.

This script demonstrates how to:
1. Set up environment variables for S3 access
2. Configure training to use S3 for data, checkpoints, and logs
3. Run training with S3 integration

Prerequisites:
- AWS credentials configured (via environment variables or AWS CLI)
- S3 bucket created and accessible
- Pokemon frames uploaded to S3 bucket

Environment Variables Required:
- AWS_ACCESS_KEY_ID: Your AWS access key
- AWS_SECRET_ACCESS_KEY: Your AWS secret key
- S3_BUCKET_NAME: Name of your S3 bucket
- AWS_REGION: AWS region (optional, defaults to us-east-1)
"""

from train import main, CONFIG
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))


def setup_s3_environment():
    """Setup environment variables for S3 access"""

    # Example environment setup - replace with your actual values
    # In production, these should be set as actual environment variables

    # Uncomment and set these if not already set in your environment:
    # os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key_here'
    # os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key_here'
    # os.environ['S3_BUCKET_NAME'] = 'your-pokemon-training-bucket'
    # os.environ['AWS_REGION'] = 'us-east-1'

    # Check if required environment variables are set
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value_here")
        print("\nOr set them in this script (not recommended for production)")
        return False

    print("✓ S3 environment variables configured")
    return True


def configure_s3_training():
    """Configure training settings for S3"""

    # S3 Configuration
    CONFIG.use_s3 = True
    CONFIG.s3_bucket = os.getenv('S3_BUCKET_NAME')
    CONFIG.s3_region = os.getenv('AWS_REGION', 'us-east-1')

    # S3 paths - adjust these based on your bucket structure
    CONFIG.frames_dir = 'pokemon_frames'  # S3 prefix where frames are stored
    CONFIG.s3_checkpoint_prefix = 'checkpoints'
    CONFIG.s3_tensorboard_prefix = 'tensorboard'
    CONFIG.s3_logs_prefix = 'logs'

    # Local cache settings
    CONFIG.local_cache_dir = '/tmp/pokemon_cache'  # Local cache for S3 images
    CONFIG.max_cache_size = 1000  # Maximum number of images to cache locally

    # Training configuration
    CONFIG.experiment_name = 'pokemon_vqvae_s3_example'
    CONFIG.batch_size = 8  # Smaller batch size for example
    CONFIG.num_epochs = 5  # Fewer epochs for example
    CONFIG.learning_rate = 1e-4
    CONFIG.save_interval = 100  # Save checkpoints more frequently
    CONFIG.eval_interval = 200
    CONFIG.log_interval = 10

    print("✓ Training configuration set for S3")


def print_configuration():
    """Print the current configuration"""
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Experiment Name: {CONFIG.experiment_name}")
    print(f"Use S3: {CONFIG.use_s3}")
    print(f"S3 Bucket: {CONFIG.s3_bucket}")
    print(f"S3 Region: {CONFIG.s3_region}")
    print(f"Frames Directory: {CONFIG.frames_dir}")
    print(f"Local Cache Dir: {CONFIG.local_cache_dir}")
    print(f"Batch Size: {CONFIG.batch_size}")
    print(f"Number of Epochs: {CONFIG.num_epochs}")
    print(f"Learning Rate: {CONFIG.learning_rate}")
    print("="*50 + "\n")


def main_example():
    """Main function for the S3 training example"""

    print("Pokemon VQVAE Training with S3 - Example Script")
    print("=" * 60)

    # Setup S3 environment
    if not setup_s3_environment():
        return 1

    # Configure training for S3
    configure_s3_training()

    # Print configuration
    print_configuration()

    # Confirm before starting
    response = input("Start training with S3? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return 0

    try:
        # Start training
        print("\nStarting S3-enabled training...")
        main()
        print("\n✓ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main_example()
    sys.exit(exit_code)
