import concurrent.futures
import logging
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from data.s3.s3_utils import default_s3_manager
from data.scraping.frame_filterer import filter_frame_sequence

sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PokemonFrameDataset(Dataset):
    """Dataset for loading consecutive Pokemon frame pairs from S3 or local storage"""

    def __init__(
        self,
        frames_dir: str,
        image_size: int = 400,
        transform=None,
        num_frames_in_video: int = 2,
        min_frame_gap: int = 1,
        max_frame_gap: int = 5,
        seed: Optional[int] = None,
        use_s3: bool = False,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 100000,
        stage: Literal["train", "test"] = "train",
        seed_cache: bool = False,
        num_concurrent_downloads: int = 50,
        limit: Optional[int] = None,
    ):
        """
        Args:
            frames_dir: Path to pokemon_frames directory (local or S3 prefix)
            image_size: Size to resize images to
            transform: Optional transform to apply to images
            min_frame_gap: Minimum gap between consecutive frames
            max_frame_gap: Maximum gap between consecutive frames
            seed: Random seed for reproducibility
            use_s3: Whether to use S3 for storage
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
        self.num_concurrent_downloads = num_concurrent_downloads
        self.limit = limit

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize S3 manager if needed
        if use_s3:
            # Setup cache directory
            if cache_dir is None:
                self.cache_dir = tempfile.mkdtemp(prefix="pokemon_frames_cache_")
            else:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)

            self.cache_usage = {}  # Track cache usage for LRU eviction
            logger.info(f"Using S3 storage with cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None

        # Find all frame pairs
        self.frame_pairs = self._find_videos(num_frames_in_video)

        if stage == "test":
            self.frame_pairs = self.frame_pairs[int(len(self.frame_pairs) * 0.95) :]
        else:
            self.frame_pairs = self.frame_pairs[: int(len(self.frame_pairs) * 0.95)]

        if limit is not None:
            self.frame_pairs = self.frame_pairs[:limit]

        if seed_cache:
            logger.info("Seeding cache...")
            unique_frame_keys = list(
                set(
                    [frame_pair[0] for frame_pair in self.frame_pairs]
                    + [frame_pair[1] for frame_pair in self.frame_pairs]
                )
            )

            self.seed_cache(unique_frame_keys)

        logger.warning(f"Found {len(self.frame_pairs)} valid frame pairs")

        # Shuffle with seed for reproducibility
        if seed is not None:
            random.Random(seed).shuffle(self.frame_pairs)

    def _find_videos(self, num_frames_in_video: int) -> List[list[str]]:
        """Find all valid consecutive frame pairs"""
        if self.use_s3:
            return self._find_videos_s3(self.frames_dir, num_frames_in_video)
        else:
            return self._find_videos_local(num_frames_in_video)

    def _find_videos_local(self, num_frames_in_video: int) -> List[list[str]]:
        """Find frame pairs in local filesystem"""
        videos = []
        logger.info(f"Finding frame pairs in {self.frames_dir}.")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.frames_dir):
            # Find all frame files in this directory
            frame_files = []
            for file in files:
                if file.startswith("frame_") and file.endswith(".png"):
                    # Extract frame number from filename
                    match = re.search(r"frame_(\d+)", file)
                    if match:
                        frame_num = int(match.group(1))
                        frame_files.append((frame_num, os.path.join(root, file)))

            # Sort by frame number
            frame_files.sort(key=lambda x: x[0])
            frame_files = [(x[0], Image.open(x[1]).convert("RGB")) for x in frame_files]
            frame_files = filter_frame_sequence(frame_files)

            # Create pairs with varying gaps
            for i in range(len(frame_files) - self.max_frame_gap):
                frame1_num, frame1_path = frame_files[i]
                if i + num_frames_in_video < len(frame_files):
                    # Verify both files exist
                    if os.path.exists(frame1_path) and all(
                        os.path.exists(frame_path)
                        for _, frame_path in frame_files[
                            i + 1 : i + num_frames_in_video
                        ]
                    ):
                        videos.append(
                            [
                                path
                                for _, path in frame_files[i : i + num_frames_in_video]
                            ]
                        )

        return videos

    def _find_videos_s3(
        self, source_dir: str, num_frames_in_video: int
    ) -> List[list[str]]:
        """Find frame pairs in S3"""
        logger.info(f"Finding frame pairs in {source_dir}")

        videos: list[list[str]] = []

        # List all objects with the frames prefix
        frame_objects = default_s3_manager.list_objects(
            prefix=source_dir, suffix=".png"
        )
        logger.info(f"Found {len(frame_objects)} frame objects")

        # Group by directory (game/episode)
        directories = {}
        for obj_key in frame_objects:
            if "frame_" in obj_key:
                # Extract directory path
                dir_path = "/".join(obj_key.split("/")[:-1])
                if dir_path not in directories:
                    directories[dir_path] = []

                # Extract frame number
                filename = obj_key.split("/")[-1]
                match = re.search(r"frame_(\d+)", filename)
                if match:
                    frame_num = int(match.group(1))
                    directories[dir_path].append((frame_num, obj_key))

        # Create pairs for each directory
        for dir_path, frame_files in tqdm(
            directories.items(), desc="Processing directories"
        ):
            # Sort by frame number
            frame_files.sort(key=lambda x: x[0])
            # Bring all frame paths to local directory

            # Create pairs with varying gaps
            for i in tqdm(
                range(0, len(frame_files) - self.max_frame_gap, num_frames_in_video),
                desc="Processing frame files",
            ):
                frame_files_to_process = frame_files[i : i + num_frames_in_video]
                frame_files_to_process = [
                    (x[1], self._load_image_from_s3(x[1]))
                    for x in frame_files_to_process
                ]
                frame_files_to_process = filter(bool, frame_files_to_process)
                frame_paths_to_process = filter_frame_sequence(frame_files_to_process)
                if len(frame_paths_to_process) < num_frames_in_video // 2:
                    continue

                # Verify both files exist
                logger.warning(f"Found {len(frame_paths_to_process)} frame files")
                videos.append(frame_paths_to_process[: num_frames_in_video // 2])

                if self.limit is not None and len(videos) >= self.limit:
                    break

        return videos

    def seed_cache(self, frames: list[str]):
        """Seed the cache with all images in the frames directory"""
        if not self.use_s3:
            raise ValueError("Cache seeding is only supported for S3 storage")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_concurrent_downloads
        ) as executor:
            futures = [
                executor.submit(self._load_image_from_s3, frame) for frame in frames
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Seeding cache",
                total=len(frames),
            ):
                future.result()

        logger.info(f"Cache seeded with {len(frames)} images")

    def _get_cached_image_path(self, s3_key: str) -> str:
        """Get local cache path for S3 image"""
        if not self.cache_dir:
            raise ValueError("Cache directory not initialized")

        # Create a safe filename from S3 key
        safe_filename = s3_key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, safe_filename)

    def _load_image_from_s3(self, s3_key: str) -> Optional[Image.Image]:
        """Load image from S3 with local caching"""
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
                except Exception as e:
                    logger.warning(
                        f"Error removing corrupted cache file {cache_path}: {e}"
                    )

        # Download from S3
        image = default_s3_manager.download_image(s3_key)
        if image is None:
            return None

        # Cache the image locally
        self._cache_image(image, cache_path, s3_key)
        return image.convert("RGB")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for training"""
        # Resize images
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))

        # Convert to tensors (C, H, W)
        # Convert PIL Image to numpy array (uint8)
        image_np = np.array(image)
        # Convert numpy array to float32 tensor and normalize to [0, 1]
        image_tensor = (
            torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )

        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

    def _cache_image(self, image: Image.Image, cache_path: str, s3_key: str):
        """Cache image locally with LRU eviction"""
        try:
            # Check cache size and evict if necessary
            if len(self.cache_usage) >= self.max_cache_size:
                self._evict_cache()

            # Save image to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image.save(cache_path, "PNG")

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
        video_path = self.frame_pairs[idx]

        # Load images based on storage type
        if self.use_s3:
            video = [self._load_image_from_s3(frame_path) for frame_path in video_path]
        else:
            video = [Image.open(frame_path).convert("RGB") for frame_path in video_path]

        if video is None:
            raise ValueError("Failed to load video")

        video_tensors = [self._preprocess_image(video) for video in video]

        out = torch.stack(video_tensors)
        logger.warning(f"Out shape [frame dataset]: {out.shape}")
        return out

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the dataset for checkpointing"""
        return {
            "seed": self.seed,
            "frames_dir": str(self.frames_dir),
            "image_size": self.image_size,
            "min_frame_gap": self.min_frame_gap,
            "max_frame_gap": self.max_frame_gap,
            "num_frame_pairs": len(self.frame_pairs),
            "use_s3": self.use_s3,
            "cache_dir": self.cache_dir,
            "max_cache_size": self.max_cache_size,
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
        if (
            hasattr(self, "cache_dir")
            and self.cache_dir
            and self.cache_dir.startswith("/tmp")
        ):
            # Only auto-cleanup temporary directories
            self.cleanup_cache()
