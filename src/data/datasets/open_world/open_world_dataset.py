import logging
import random
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from data.datasets.cache import Cache
from data.datasets.data_types.open_world_types import (
    OpenWorldVideoLog,
    OpenWorldVideoLogSingleton,
)
from data.scraping.frame_filterer import get_frame_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenWorldRunningDataset(Dataset):
    """Dataset of pre-extracted sliding-window video clips.

    Sliding windows are baked into the video logs at dataset-creation time by
    `OpenWorldRunningDatasetCreator` (each `OpenWorldVideoLogSingleton` is
    already a single window of exactly `num_images_in_video` frames, sampled
    with the configured frame_spacing). This dataset therefore just shuffles
    the pre-extracted windows and optionally filters them by distinct-frame
    count.
    """

    dataset: OpenWorldVideoLog

    def __init__(
        self,
        dataset: OpenWorldVideoLog,
        local_cache: Cache,
        image_size: int,
        num_images_in_video: int,
        num_unique_frames: int | None = None,
        limit: int | None = None,
    ):
        if limit is not None:
            dataset = OpenWorldVideoLog(video_logs=dataset.video_logs[:limit])

        self.dataset = dataset
        self.local_cache = local_cache
        self.image_size = image_size
        self.num_images_in_video = num_images_in_video
        self.num_unique_frames = num_unique_frames

        self.samples = self._build_sample_index()
        logger.info(
            f"Using {len(self.samples)} pre-extracted window samples from "
            f"{len(dataset.video_logs)} video logs "
            f"(num_images_in_video={num_images_in_video})"
        )

    def _count_unique_frames(self, window: OpenWorldVideoLogSingleton) -> int:
        """Load the frames for a window and count how many are distinct."""
        images = [self.local_cache.get(p) for p in window.video_log_paths]

        unique = 1
        for i in range(1, len(images)):
            if images[i - 1] is None or images[i] is None:
                continue
            if bool(get_frame_similarity(images[i - 1], images[i]) <= 0.8):  # type: ignore[arg-type]
                unique += 1
        return unique

    def _build_sample_index(self) -> list[OpenWorldVideoLogSingleton]:
        """Return the (optionally filtered and shuffled) list of window samples."""
        samples = [
            v for v in self.dataset.video_logs
            if len(v.video_log_paths) >= self.num_images_in_video
        ]

        if self.num_unique_frames is not None:
            pre_filter_count = len(samples)
            samples = [
                s for s in tqdm(samples, desc="Filtering by unique frames")
                if self._count_unique_frames(s) >= self.num_unique_frames  # type: ignore[operator]
            ]
            logger.info(
                f"Unique-frame filter: {pre_filter_count} -> {len(samples)} samples "
                f"(num_unique_frames={self.num_unique_frames})"
            )

        rng = random.Random(42)
        rng.shuffle(samples)
        return samples

    def _load_image_locally(self, frame_path: str) -> Image.Image | None:
        return self.local_cache.get(frame_path)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))

        image_np = np.array(image)
        image_tensor = (
            torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )

        return image_tensor

    #### BASIC DATASET METHODS ####

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        video_paths = self.samples[idx].video_log_paths

        try:
            images = [self._load_image_locally(path) for path in video_paths]
            if any(img is None for img in images):
                raise ValueError("Failed to load video frame from cache")

            video_tensors = [self._preprocess_image(img) for img in images]  # type: ignore[arg-type]
            out = torch.stack(video_tensors)
            logger.debug(f"Out shape: {out.shape}")
            return out
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error loading video: {e}")
            return None
