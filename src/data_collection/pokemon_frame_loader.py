#!/usr/bin/env python3
"""
Pokemon Frame Loader for Training
Loads consecutive frame pairs from Pokemon gameplay videos for training.
Supports lazy loading, batch generation, resumable training, and S3 storage.
"""

from data_collection.frame_dataset import PokemonFrameDataset
from data_collection.resumable_data_loader import ResumableDataLoader
from s3.s3_utils import S3Manager, get_s3_manager_from_env
import os
from pathlib import Path
from typing import Literal, Tuple, Iterator, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
import logging
import json
from s3.s3_utils import default_s3_manager

# Add the parent directory to the path so we can import from idm
import sys
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


class PokemonFrameLoader:
    """Main loader class for Pokemon frames with resumable training support and S3 integration"""

    def __init__(
        self,
        frames_dir: str,
        num_frames_in_video: int = 2,
        batch_size: int = 8,
        image_size: int = 400,
        shuffle: bool = True,
        num_workers: int = 4,
        min_frame_gap: int = 1,
        max_frame_gap: int = 5,
        seed: Optional[int] = None,
        use_s3: bool = False,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 100000,
        stage: Literal["train", "test"] = "train",
        seed_cache: bool = False,
        limit: Optional[int] = None
    ):
        """
        Args:
            frames_dir: Path to pokemon_frames directory (local or S3 prefix)
            batch_size: Batch size for training
            image_size: Size to resize images to
            shuffle: Whether to shuffle the dataset
            num_workers: Number of workers for data loading
            min_frame_gap: Minimum gap between consecutive frames
            max_frame_gap: Maximum gap between consecutive frames
            seed: Random seed for reproducibility
            use_s3: Whether to use S3 for storage
            cache_dir: Local cache directory for S3 images
            max_cache_size: Maximum number of images to cache locally
        """
        self.frames_dir = frames_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.use_s3 = use_s3
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size

        # Create dataset
        self.dataset = PokemonFrameDataset(
            frames_dir=frames_dir,
            image_size=image_size,
            num_frames_in_video=num_frames_in_video,
            min_frame_gap=min_frame_gap,
            max_frame_gap=max_frame_gap,
            seed=seed,
            use_s3=use_s3,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            stage=stage,
            seed_cache=seed_cache,
            limit=limit
        )

        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Create resumable wrapper
        self.resumable_loader = ResumableDataLoader(self.dataloader)

    def create_resumable_loader(self, start_epoch: int = 0, start_batch: int = 0) -> ResumableDataLoader:
        """Create a resumable data loader starting from specific epoch/batch"""
        return ResumableDataLoader(self.dataloader, start_epoch, start_batch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate through batches of frame pairs"""
        return iter(self.resumable_loader)

    def __len__(self) -> int:
        """Number of batches"""
        return len(self.dataloader)

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample batch for testing"""
        return next(iter(self.dataloader))

    def get_dataset_info(self) -> dict:
        """Get information about the dataset"""
        return {
            'total_frame_pairs': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': len(self.dataloader),
            'image_size': self.image_size,
            'frames_directory': str(self.frames_dir),
            'seed': self.seed,
            'use_s3': self.use_s3,
            'cache_dir': self.cache_dir,
            'max_cache_size': self.max_cache_size
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing"""
        return {
            'frames_dir': self.frames_dir,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'use_s3': self.use_s3,
            'cache_dir': self.cache_dir,
            'max_cache_size': self.max_cache_size,
            'dataset_state': self.dataset.get_state(),
            'loader_state': self.resumable_loader.get_state()
        }

    def save_state(self, filepath: str):
        """Save the current state to a file or S3"""
        state = self.get_state()

        if self.use_s3:
            # Save to S3
            success = default_s3_manager.upload_json(state, filepath)
            if success:
                logger.info(f"DataLoader state saved to S3: {filepath}")
            else:
                logger.error(f"Failed to save DataLoader state to S3: {filepath}")
        else:
            # Save locally
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"DataLoader state saved to: {filepath}")

    @classmethod
    def load_state(cls, filepath: str, s3_manager: Optional[S3Manager] = None) -> 'PokemonFrameLoader':
        """Load state from a file or S3 and create a new loader"""

        # Determine if we should load from S3
        use_s3_for_state = filepath.startswith('s3://') or (s3_manager is not None and not os.path.exists(filepath))

        if use_s3_for_state:
            if s3_manager is None:
                s3_manager = get_s3_manager_from_env()
            state = s3_manager.download_json(filepath)
            if state is None:
                raise ValueError(f"Failed to load state from S3: {filepath}")
        else:
            with open(filepath, 'r') as f:
                state = json.load(f)

        loader = cls(
            frames_dir=state['frames_dir'],
            batch_size=state['batch_size'],
            image_size=state['image_size'],
            shuffle=state['shuffle'],
            num_workers=state['num_workers'],
            seed=state['seed'],
            use_s3=state.get('use_s3', False),
            cache_dir=state.get('cache_dir'),
            max_cache_size=state.get('max_cache_size', 1000)
        )

        # Restore loader state
        loader_state = state['loader_state']
        loader.resumable_loader = ResumableDataLoader(
            loader.dataloader,
            start_epoch=loader_state['current_epoch'],
            start_batch=loader_state['current_batch']
        )

        logger.info(f"DataLoader state loaded from {filepath}")
        return loader

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.dataset, 'cleanup_cache'):
            self.dataset.cleanup_cache()


def main():
    """Test the frame loader"""
    import argparse

    parser = argparse.ArgumentParser(description='Pokemon Frame Loader')
    parser.add_argument('--frames-dir', default='pokemon_frames', help='Path to frames directory or S3 prefix')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--image-size', type=int, default=400, help='Image size')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-s3', action='store_true', help='Use S3 storage')
    parser.add_argument('--cache-dir', help='Local cache directory for S3 images')

    args = parser.parse_args()

    # Create loader
    loader = PokemonFrameLoader(
        frames_dir=args.frames_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        use_s3=args.use_s3,
        cache_dir=args.cache_dir
    )

    # Print info
    info = loader.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if args.test:
        print("\nTesting data loading...")
        try:
            batch1, batch2 = loader.get_sample_batch()
            print(f"Batch 1 shape: {batch1.shape}")
            print(f"Batch 2 shape: {batch2.shape}")
            print(f"Batch 1 dtype: {batch1.dtype}")
            print(f"Batch 2 dtype: {batch2.dtype}")
            print(f"Batch 1 range: [{batch1.min():.3f}, {batch1.max():.3f}]")
            print(f"Batch 2 range: [{batch2.min():.3f}, {batch2.max():.3f}]")
            print("✓ Data loading successful!")

            # Test state saving/loading
            print("\nTesting state save/load...")
            state_path = "test_loader_state.json"
            if args.use_s3:
                state_path = "test_states/test_loader_state.json"

            loader.save_state(state_path)
            loaded_loader = PokemonFrameLoader.load_state(state_path)
            print("✓ State save/load successful!")

        except Exception as e:
            print(f"✗ Error loading data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            loader.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
