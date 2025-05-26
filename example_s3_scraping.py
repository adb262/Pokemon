#!/usr/bin/env python3
"""
Example script for running Pokemon dataset pipeline with S3 storage.

This script demonstrates how to:
1. Set up environment variables for S3 access
2. Configure the pipeline to use S3 for data storage
3. Run the complete pipeline with S3 integration

Prerequisites:
- AWS credentials configured (via environment variables or AWS CLI)
- S3 bucket created and accessible
- Required Python packages installed

Environment Variables Required:
- AWS_ACCESS_KEY_ID: Your AWS access key
- AWS_SECRET_ACCESS_KEY: Your AWS secret key
- S3_BUCKET_NAME: Name of your S3 bucket
- AWS_REGION: AWS region (optional, defaults to us-east-1)
"""

from data_collection.data_config import PokemonDatasetPipelineConfig
from data_collection.pokemon_dataset_pipeline import PokemonDatasetPipeline
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
    # os.environ['S3_BUCKET_NAME'] = 'your-pokemon-dataset-bucket'
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


def configure_s3_pipeline():
    """Configure pipeline settings for S3"""

    config = PokemonDatasetPipelineConfig()

    # S3 Configuration
    config.use_s3 = True
    config.s3_raw_prefix = 'raw_videos'
    config.s3_clean_prefix = 'clean_videos'
    config.s3_frames_prefix = 'pokemon_frames'

    # Local storage settings
    config.keep_local_videos = False  # Don't keep local video copies to save space
    config.keep_local_frames = True   # Keep local frames for faster access during training

    # Pipeline settings
    config.max_videos_per_game = 10   # Smaller number for example
    config.target_fps = 5
    config.target_height = 360
    config.min_gameplay_ratio = 0.6

    # Local directories (used as temporary/cache directories when using S3)
    config.raw_videos_dir = 'temp_raw_videos'
    config.clean_videos_dir = 'temp_clean_videos'
    config.frames_dir = 'pokemon_frames'
    config.logs_dir = 'logs'

    # Pipeline steps
    config.scrape = True
    config.clean = True
    config.extract = True
    config.summary = True

    print("✓ Pipeline configuration set for S3")
    return config


def print_configuration(config: PokemonDatasetPipelineConfig):
    """Print the current configuration"""
    print("\n" + "="*60)
    print("POKEMON DATASET PIPELINE CONFIGURATION")
    print("="*60)
    print(f"Use S3: {config.use_s3}")
    print(f"S3 Bucket: {os.getenv('S3_BUCKET_NAME')}")
    print(f"S3 Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    print(f"S3 Raw Videos Prefix: {config.s3_raw_prefix}")
    print(f"S3 Clean Videos Prefix: {config.s3_clean_prefix}")
    print(f"S3 Frames Prefix: {config.s3_frames_prefix}")
    print(f"Keep Local Videos: {config.keep_local_videos}")
    print(f"Keep Local Frames: {config.keep_local_frames}")
    print(f"Max Videos per Game: {config.max_videos_per_game}")
    print(f"Target FPS: {config.target_fps}")
    print(f"Target Height: {config.target_height}")
    print(
        f"Pipeline Steps: Scrape={config.scrape}, Clean={config.clean}, Extract={config.extract}, Summary={config.summary}")
    print("="*60 + "\n")


def main_example():
    """Main function for the S3 pipeline example"""

    print("Pokemon Dataset Pipeline with S3 - Example Script")
    print("=" * 60)

    # Setup S3 environment
    if not setup_s3_environment():
        return 1

    # Configure pipeline for S3
    config = configure_s3_pipeline()

    # Print configuration
    print_configuration(config)

    # Confirm before starting
    response = input("Start pipeline with S3? (y/N): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return 0

    try:
        # Create and run pipeline
        print("\nStarting S3-enabled pipeline...")
        pipeline = PokemonDatasetPipeline(config)

        # Print pipeline info
        pipeline_info = pipeline.get_pipeline_info()
        print("\nPipeline Info:")
        for key, value in pipeline_info.items():
            print(f"  {key}: {value}")

        # Run the pipeline
        success = pipeline.run_full_pipeline()

        if success:
            print("\n✓ Pipeline completed successfully!")
            print("\nYour Pokemon dataset is now available in S3!")
            print(f"Bucket: {os.getenv('S3_BUCKET_NAME')}")
            print(f"Frames: s3://{os.getenv('S3_BUCKET_NAME')}/{config.s3_frames_prefix}/")
        else:
            print("\n✗ Pipeline failed!")
            return 1

    except KeyboardInterrupt:
        print("\n⚠ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def run_individual_steps():
    """Example of running individual pipeline steps"""

    print("\nExample: Running individual pipeline steps")
    print("-" * 50)

    if not setup_s3_environment():
        return 1

    config = configure_s3_pipeline()
    pipeline = PokemonDatasetPipeline(config)

    # Run steps individually
    steps = [
        ("Scraping videos", pipeline.step1_scrape_videos),
        ("Cleaning videos", pipeline.step2_clean_videos),
        ("Extracting frames", pipeline.step3_extract_frames),
        ("Creating summary", pipeline.step4_create_dataset_summary)
    ]

    for step_name, step_func in steps:
        response = input(f"Run {step_name}? (y/N): ")
        if response.lower() == 'y':
            print(f"\nRunning {step_name}...")
            success = step_func()
            if success:
                print(f"✓ {step_name} completed successfully")
            else:
                print(f"✗ {step_name} failed")
                return 1
        else:
            print(f"Skipping {step_name}")

    print("\n✓ Individual steps completed!")
    return 0


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run full pipeline")
    print("2. Run individual steps")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        exit_code = main_example()
    elif choice == "2":
        exit_code = run_individual_steps()
    else:
        print("Invalid choice")
        exit_code = 1

    sys.exit(exit_code)
