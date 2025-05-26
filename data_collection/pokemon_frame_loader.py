#!/usr/bin/env python3
"""
Pokemon Frame Loader for Training
Loads consecutive frame pairs from Pokemon gameplay videos for training.
Supports lazy loading, batch generation, resumable training, and S3 storage.
"""

from idm.s3_utils import S3Manager, get_s3_manager_from_env
import os
import re
import random
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import logging
import json
import concurrent.futures

# Add the parent directory to the path so we can import from idm
import sys
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


class PokemonFrameDataset(Dataset):
    """Dataset for loading consecutive Pokemon frame pairs from S3 or local storage"""

    def __init__(self,
                 frames_dir: str,
                 image_size: int = 400,
                 transform=None,
                 min_frame_gap: int = 1,
                 max_frame_gap: int = 5,
                 seed: Optional[int] = None,
                 use_s3: bool = False,
                 s3_manager: Optional[S3Manager] = None,
                 cache_dir: Optional[str] = None,
                 max_cache_size: int = 1000):
        """
        Args:
            frames_dir: Path to pokemon_frames directory (local or S3 prefix)
            image_size: Size to resize images to
            transform: Optional transform to apply to images
            min_frame_gap: Minimum gap between consecutive frames
            max_frame_gap: Maximum gap between consecutive frames
            seed: Random seed for reproducibility
            use_s3: Whether to use S3 for storage
            s3_manager: S3Manager instance (if None and use_s3=True, will create from env)
            cache_dir: Local cache directory for S3 images
            max_cache_size: Maximum number of images to cache locally
        """
        self.frames_dir = frames_dir
        self.image_size = image_size
        self.transform = transform
        self.min_frame_gap = min_frame_gap
        self.max_frame_gap = max_frame_gap
        self.seed = seed
        self.use_s3 = use_s3
        self.max_cache_size = max_cache_size

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize S3 manager if needed
        if use_s3:
            if s3_manager is None:
                self.s3_manager = get_s3_manager_from_env()
            else:
                self.s3_manager = s3_manager

            # Setup cache directory
            if cache_dir is None:
                self.cache_dir = tempfile.mkdtemp(prefix="pokemon_frames_cache_")
            else:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)

            self.cache_usage = {}  # Track cache usage for LRU eviction
            logger.info(f"Using S3 storage with cache directory: {self.cache_dir}")
        else:
            self.s3_manager = None
            self.cache_dir = None

        # Find all frame pairs
        self.frame_pairs = self._find_frame_pairs()
        logger.info(f"Found {len(self.frame_pairs)} valid frame pairs")

        # Shuffle with seed for reproducibility
        if seed is not None:
            random.Random(seed).shuffle(self.frame_pairs)

    def _find_frame_pairs(self) -> List[Tuple[str, str]]:
        """Find all valid consecutive frame pairs"""
        if self.use_s3:
            return self._find_frame_pairs_s3()
        else:
            return self._find_frame_pairs_local()

    def _find_frame_pairs_local(self) -> List[Tuple[str, str]]:
        """Find frame pairs in local filesystem"""
        frame_pairs = []

        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.frames_dir):
            # Find all frame files in this directory
            frame_files = []
            for file in files:
                if file.startswith('frame_') and file.endswith('.png'):
                    # Extract frame number from filename
                    match = re.search(r'frame_(\d+)', file)
                    if match:
                        frame_num = int(match.group(1))
                        frame_files.append((frame_num, os.path.join(root, file)))

            # Sort by frame number
            frame_files.sort(key=lambda x: x[0])

            # Create pairs with varying gaps
            for i in range(len(frame_files) - self.max_frame_gap):
                frame1_num, frame1_path = frame_files[i]

                # Try different gaps within the specified range
                for gap in range(self.min_frame_gap, min(self.max_frame_gap + 1, len(frame_files) - i)):
                    if i + gap < len(frame_files):
                        frame2_num, frame2_path = frame_files[i + gap]

                        # Verify both files exist
                        if os.path.exists(frame1_path) and os.path.exists(frame2_path):
                            frame_pairs.append((frame1_path, frame2_path))

        return frame_pairs

    def _find_frame_pairs_s3(self) -> List[Tuple[str, str]]:
        """Find frame pairs in S3"""
        if not self.s3_manager:
            raise ValueError("S3Manager not initialized")

        frame_pairs = []

        # List all objects with the frames prefix
        frame_objects = self.s3_manager.list_objects(
            prefix=self.frames_dir,
            suffix='.png'
        )

        # Group by directory (game/episode)
        directories = {}
        for obj_key in frame_objects:
            if 'frame_' in obj_key:
                # Extract directory path
                dir_path = '/'.join(obj_key.split('/')[:-1])
                if dir_path not in directories:
                    directories[dir_path] = []

                # Extract frame number
                filename = obj_key.split('/')[-1]
                match = re.search(r'frame_(\d+)', filename)
                if match:
                    frame_num = int(match.group(1))
                    directories[dir_path].append((frame_num, obj_key))

        # Create pairs for each directory
        for dir_path, frame_files in directories.items():
            # Sort by frame number
            frame_files.sort(key=lambda x: x[0])

            # Create pairs with varying gaps
            for i in range(len(frame_files) - self.max_frame_gap):
                frame1_num, frame1_key = frame_files[i]

                # Try different gaps within the specified range
                for gap in range(self.min_frame_gap, min(self.max_frame_gap + 1, len(frame_files) - i)):
                    if i + gap < len(frame_files):
                        frame2_num, frame2_key = frame_files[i + gap]
                        frame_pairs.append((frame1_key, frame2_key))

        return frame_pairs

    def seed_cache_with_everything(self):
        """Seed the cache with all images in the frames directory"""
        if not self.use_s3:
            raise ValueError("Cache seeding is only supported for S3 storage")

        if not self.s3_manager:
            raise ValueError("S3Manager not initialized")

        # List all objects with the frames prefix
        frame_objects = self.s3_manager.list_objects(
            prefix=self.frames_dir,
            suffix='.png'
        )

        logger.info(f"Seeding cache with {len(frame_objects)} images")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self._load_image_from_s3, frame_objects)

        logger.info(f"Cache seeded with {len(frame_objects)} images")

    def _get_cached_image_path(self, s3_key: str) -> str:
        """Get local cache path for S3 image"""
        if not self.cache_dir:
            raise ValueError("Cache directory not initialized")

        # Create a safe filename from S3 key
        safe_filename = s3_key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, safe_filename)

    def _load_image_from_s3(self, s3_key: str) -> Optional[Image.Image]:
        """Load image from S3 with local caching"""
        if not self.s3_manager:
            raise ValueError("S3Manager not initialized")

        cache_path = self._get_cached_image_path(s3_key)

        # Check if image is already cached
        if os.path.exists(cache_path):
            try:
                image = Image.open(cache_path).convert("RGB")
                # Update cache usage
                self.cache_usage[s3_key] = self.cache_usage.get(s3_key, 0) + 1
                return image
            except Exception as e:
                logger.warning(f"Error loading cached image {cache_path}: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass

        # Download from S3
        try:
            image = self.s3_manager.download_image(s3_key)
            if image is None:
                return None

            # Cache the image locally
            self._cache_image(image, cache_path, s3_key)
            return image.convert("RGB")

        except Exception as e:
            logger.error(f"Error loading image from S3 {s3_key}: {e}")
            return None

    def _cache_image(self, image: Image.Image, cache_path: str, s3_key: str):
        """Cache image locally with LRU eviction"""
        try:
            # Check cache size and evict if necessary
            if len(self.cache_usage) >= self.max_cache_size:
                self._evict_cache()

            # Save image to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image.save(cache_path, 'PNG')

            # Update cache usage
            self.cache_usage[s3_key] = 1

        except Exception as e:
            logger.warning(f"Error caching image {cache_path}: {e}")

    def _evict_cache(self):
        """Evict least recently used cache entries"""
        # Sort by usage count (ascending)
        sorted_usage = sorted(self.cache_usage.items(), key=lambda x: x[1])

        # Remove bottom 20% of cache
        num_to_remove = max(1, len(sorted_usage) // 5)

        for s3_key, _ in sorted_usage[:num_to_remove]:
            cache_path = self._get_cached_image_path(s3_key)
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                del self.cache_usage[s3_key]
            except Exception as e:
                logger.warning(f"Error evicting cache file {cache_path}: {e}")

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a pair of consecutive frames"""
        frame1_path, frame2_path = self.frame_pairs[idx]

        try:
            # Load images based on storage type
            if self.use_s3:
                image1 = self._load_image_from_s3(frame1_path)
                image2 = self._load_image_from_s3(frame2_path)
            else:
                image1 = Image.open(frame1_path).convert("RGB")
                image2 = Image.open(frame2_path).convert("RGB")

            if image1 is None or image2 is None:
                raise ValueError("Failed to load one or both images")

            # Resize images
            image1 = image1.resize((self.image_size, self.image_size))
            image2 = image2.resize((self.image_size, self.image_size))

            # Convert to tensors
            image1_tensor = torch.tensor(np.array(image1), dtype=torch.float32).permute(2, 0, 1) / 255.0
            image2_tensor = torch.tensor(np.array(image2), dtype=torch.float32).permute(2, 0, 1) / 255.0

            # Apply transforms if provided
            if self.transform:
                image1_tensor = self.transform(image1_tensor)
                image2_tensor = self.transform(image2_tensor)

            return image1_tensor, image2_tensor

        except Exception as e:
            logger.warning(f"Error loading frame pair {frame1_path}, {frame2_path}: {e}")
            # Return a random different pair if this one fails
            new_idx = random.randint(0, len(self.frame_pairs) - 1)
            return self.__getitem__(new_idx)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the dataset for checkpointing"""
        return {
            'seed': self.seed,
            'frames_dir': str(self.frames_dir),
            'image_size': self.image_size,
            'min_frame_gap': self.min_frame_gap,
            'max_frame_gap': self.max_frame_gap,
            'num_frame_pairs': len(self.frame_pairs),
            'use_s3': self.use_s3,
            'cache_dir': self.cache_dir,
            'max_cache_size': self.max_cache_size
        }

    def cleanup_cache(self):
        """Clean up cache directory"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up cache directory: {e}")

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        if hasattr(self, 'cache_dir') and self.cache_dir and self.cache_dir.startswith('/tmp'):
            # Only auto-cleanup temporary directories
            self.cleanup_cache()


class ResumableDataLoader:
    """A wrapper around DataLoader that supports resumable training"""

    def __init__(self, dataloader: DataLoader, start_epoch: int = 0, start_batch: int = 0):
        self.dataloader = dataloader
        self.start_epoch = start_epoch
        self.start_batch = start_batch
        self.current_epoch = start_epoch
        self.current_batch = 0

    def __iter__(self):
        """Iterate through the dataloader, skipping to the correct position if resuming"""
        self.current_batch = 0

        # If we're resuming from a specific batch, skip ahead
        if self.current_epoch == self.start_epoch and self.start_batch > 0:
            logger.info(f"Resuming from epoch {self.start_epoch}, batch {self.start_batch}")

            # Skip batches
            iterator = iter(self.dataloader)
            for _ in range(self.start_batch):
                try:
                    next(iterator)
                    self.current_batch += 1
                except StopIteration:
                    break

            # Continue from where we left off
            for batch in iterator:
                yield batch
                self.current_batch += 1
        else:
            # Normal iteration
            for batch in self.dataloader:
                yield batch
                self.current_batch += 1

    def __len__(self):
        return len(self.dataloader)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing"""
        return {
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'dataset_state': {}  # We'll handle this at the PokemonFrameLoader level
        }

    def set_epoch(self, epoch: int):
        """Set the current epoch"""
        self.current_epoch = epoch
        self.current_batch = 0


class PokemonFrameLoader:
    """Main loader class for Pokemon frames with resumable training support and S3 integration"""

    def __init__(self,
                 frames_dir: str,
                 batch_size: int = 8,
                 image_size: int = 400,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 min_frame_gap: int = 1,
                 max_frame_gap: int = 5,
                 seed: Optional[int] = None,
                 use_s3: bool = False,
                 s3_manager: Optional[S3Manager] = None,
                 cache_dir: Optional[str] = None,
                 max_cache_size: int = 1000):
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
            s3_manager: S3Manager instance
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
        self.s3_manager = s3_manager
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size

        # Create dataset
        self.dataset = PokemonFrameDataset(
            frames_dir=frames_dir,
            image_size=image_size,
            min_frame_gap=min_frame_gap,
            max_frame_gap=max_frame_gap,
            seed=seed,
            use_s3=use_s3,
            s3_manager=s3_manager,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size
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

        if self.use_s3 and self.s3_manager:
            # Save to S3
            success = self.s3_manager.upload_json(state, filepath)
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
            s3_manager=s3_manager,
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
            loaded_loader = PokemonFrameLoader.load_state(state_path, loader.s3_manager)
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
