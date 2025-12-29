#!/usr/bin/env python3
"""
Pokemon Frame Loader for Training
Loads consecutive frame pairs from Pokemon gameplay videos for training.
Supports lazy loading, batch generation, resumable training, and S3 storage.
"""

import logging

# Add the parent directory to the path so we can import from idm
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader

from data.data_loaders.resumable_data_loader import ResumableDataLoader
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset

sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


class PokemonOpenWorldLoader:
    """Main loader class for Pokemon frames with resumable training support and S3 integration"""

    def __init__(
        self,
        frames_dir: str,
        dataset: OpenWorldRunningDataset,
        batch_size: int = 8,
        image_size: int = 400,
        shuffle: bool = True,
        num_workers: int = 4,
        seed: Optional[int] = None,
        use_s3: bool = False,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 100000,
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
        self.dataset = dataset

        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False if torch.backends.mps.is_available() else True,
            drop_last=True,
            persistent_workers=True,
        )

        # Create resumable wrapper
        self.resumable_loader = ResumableDataLoader(self.dataloader)

    def create_resumable_loader(
        self, start_epoch: int = 0, start_batch: int = 0
    ) -> ResumableDataLoader:
        """Create a resumable data loader starting from specific epoch/batch"""
        return ResumableDataLoader(self.dataloader, start_epoch, start_batch)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through batches of frame pairs"""
        return iter(self.resumable_loader)

    def __len__(self) -> int:
        """Number of batches"""
        return len(self.dataloader)

    def get_sample_batch(self) -> torch.Tensor:
        """Get a single sample batch for testing"""
        return next(iter(self.dataloader))

    def get_dataset_info(self) -> dict:
        """Get information about the dataset"""
        return {
            "total_frame_pairs": len(self.dataset),
            "batch_size": self.batch_size,
            "num_batches": len(self.dataloader),
            "image_size": self.image_size,
            "frames_directory": str(self.frames_dir),
            "seed": self.seed,
            "use_s3": self.use_s3,
            "cache_dir": self.cache_dir,
            "max_cache_size": self.max_cache_size,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing"""
        return {
            "frames_dir": self.frames_dir,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "use_s3": self.use_s3,
            "cache_dir": self.cache_dir,
            "max_cache_size": self.max_cache_size,
            "loader_state": self.resumable_loader.get_state(),
        }
