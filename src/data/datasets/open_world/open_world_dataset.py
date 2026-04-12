import logging
import random
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from data.datasets.cache import Cache
from data.datasets.data_types.open_world_types import (
    OpenWorldVideoLog,
)
from data.scraping.frame_filterer import get_frame_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SlidingWindowSample:
    video_log_index: int
    start_offset: int


class OpenWorldRunningDataset(Dataset):
    dataset: OpenWorldVideoLog

    def __init__(
        self,
        dataset: OpenWorldVideoLog,
        local_cache: Cache,
        image_size: int,
        num_images_in_video: int,
        frame_spacing: int = 1,
        num_unique_frames: int | None = None,
        limit: int | None = None,
    ):
        if limit is not None:
            dataset = OpenWorldVideoLog(video_logs=dataset.video_logs[:limit])

        self.dataset = dataset
        self.local_cache = local_cache
        self.image_size = image_size
        self.num_images_in_video = num_images_in_video
        self.frame_spacing = frame_spacing
        self.num_unique_frames = num_unique_frames

        self.samples = self._build_sliding_window_index()
        logger.info(
            f"Built {len(self.samples)} sliding-window samples from "
            f"{len(dataset.video_logs)} video sequences "
            f"(num_images_in_video={num_images_in_video}, frame_spacing={frame_spacing})"
        )

    def _get_frame_paths_for_window(
        self, video_paths: list[str], start_offset: int
    ) -> list[str]:
        return [
            video_paths[i]
            for i in range(
                start_offset,
                start_offset + self.num_images_in_video * self.frame_spacing,
                self.frame_spacing,
            )
        ]

    def _count_unique_frames_for_sample(self, sample: SlidingWindowSample) -> int:
        """Load the frames for a sample and count how many are distinct."""
        paths = self._get_frame_paths_for_window(
            self.dataset.video_logs[sample.video_log_index].video_log_paths,
            sample.start_offset,
        )
        images = [self.local_cache.get(p) for p in paths]

        unique = 1
        for i in range(1, len(images)):
            if images[i - 1] is None or images[i] is None:
                continue
            if bool(get_frame_similarity(images[i - 1], images[i]) <= 0.8):  # type: ignore[arg-type]
                unique += 1
        return unique

    def _build_sliding_window_index(self) -> list[SlidingWindowSample]:
        """Build a shuffled flat index of sliding window samples.

        1. Generate all valid sliding windows across every video sequence.
        2. If num_unique_frames is set, filter out windows with too few distinct frames.
        3. Shuffle the result for training data mixture.
        """
        span = (self.num_images_in_video - 1) * self.frame_spacing + 1

        samples: list[SlidingWindowSample] = []
        for video_idx, video_log in enumerate(self.dataset.video_logs):
            n = len(video_log.video_log_paths)
            num_windows = n - span + 1
            if num_windows <= 0:
                continue
            samples.extend(
                SlidingWindowSample(video_log_index=video_idx, start_offset=offset)
                for offset in range(num_windows)
            )

        if self.num_unique_frames is not None:
            pre_filter_count = len(samples)
            samples = [
                s for s in tqdm(samples, desc="Filtering by unique frames")
                if self._count_unique_frames_for_sample(s) >= self.num_unique_frames  # type: ignore[operator]
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
        sample = self.samples[idx]
        all_paths = self.dataset.video_logs[sample.video_log_index].video_log_paths
        video_paths = self._get_frame_paths_for_window(all_paths, sample.start_offset)

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
