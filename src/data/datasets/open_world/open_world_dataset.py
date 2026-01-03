import logging
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.datasets.cache import Cache
from data.datasets.data_types.open_world_types import (
    OpenWorldVideoLog,
)

random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenWorldRunningDataset(Dataset):
    dataset: OpenWorldVideoLog

    def __init__(
        self,
        dataset: OpenWorldVideoLog,
        local_cache: Cache,
        image_size: int,
        limit: int | None = None,
    ):
        if limit is not None:
            dataset = OpenWorldVideoLog(video_logs=dataset.video_logs[:limit])

        self.dataset = dataset
        self.local_cache = local_cache
        self.image_size = image_size

    def _load_image_locally(self, frame_path: str) -> Image.Image | None:
        return self.local_cache.get(frame_path)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for training"""
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))

        # Convert to tensors (C, H, W)
        # Convert PIL Image to numpy array (uint8)
        image_np = np.array(image)
        # Convert numpy array to float32 tensor and normalize to [0, 1]
        image_tensor = (
            torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )

        return image_tensor

    #### BASIC DATASET METHODS ####

    def __len__(self) -> int:
        """Number of batches"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")

        return len(self.dataset.video_logs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.dataset is None:
            raise ValueError("Dataset not loaded")

        video_paths = self.dataset.video_logs[idx].video_log_paths

        # At this point, all images are loaded
        video = [self._load_image_locally(frame_path) for frame_path in video_paths]
        if any(video is None for video in video):
            raise ValueError("Failed to load video")

        video_tensors = [self._preprocess_image(x) for x in video if x is not None]

        out = torch.stack(video_tensors)
        logger.debug(f"Out shape: {out.shape}")

        return out
