import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from typing import Literal

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.datasets.cache import Cache
from data.datasets.data_types.open_world_types import (
    OpenWorldVideoLog,
    OpenWorldVideoLogSingleton,
)
from data.scraping.frame_filterer import FrameWithPath, filter_frame_sequence
from s3.gather_frames_in_dirs import S3Frame, list_frames_in_s3
from s3.s3_utils import S3Manager, default_s3_manager

random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenWorldRunningDatasetCreator:
    max_frame_sequence_length: int = 100
    frame_chunk_size: int = 1000

    def __init__(
        self,
        dataset_dir: str,
        num_frames_in_video: int,
        output_log_json_file_name: str,
        local_cache: Cache,
        limit: int,
        image_size: int,
    ):
        self.dataset_dir = dataset_dir
        self.num_frames_in_video = num_frames_in_video
        self.output_log_dir = output_log_json_file_name
        self.local_cache = local_cache
        self.limit = limit
        self.image_size = image_size

    def _get_or_create_dataset(
        self, train_percentage: float = 0.9
    ) -> tuple[OpenWorldVideoLog, OpenWorldVideoLog]:
        train_dataset_key = self._get_dataset_key("train", train_percentage)
        test_dataset_key = self._get_dataset_key("test", train_percentage)
        logger.info(f"Train dataset key: {train_dataset_key}")
        logger.info(f"Test dataset key: {test_dataset_key}")
        if os.path.exists(train_dataset_key) and os.path.exists(test_dataset_key):
            return (
                self._load_existing_dataset(train_dataset_key),
                self._load_existing_dataset(test_dataset_key),
            )

        data = self._pull_dataset_from_s3(self.dataset_dir)
        dataset = OpenWorldVideoLog(video_logs=data)
        train, test = train_test_split(
            dataset.video_logs, test_size=1 - train_percentage, random_state=42
        )
        train_dataset = OpenWorldVideoLog(video_logs=train)
        test_dataset = OpenWorldVideoLog(video_logs=test)

        self._save_dataset(train_dataset, train_dataset_key)
        self._save_dataset(
            test_dataset,
            test_dataset_key,
        )

        return train_dataset, test_dataset

    def _filter_to_valid_videos(
        self, dataset: OpenWorldVideoLog, limit: int
    ) -> OpenWorldVideoLog:
        valid_videos: list[OpenWorldVideoLogSingleton] = []
        for video in dataset.video_logs:
            if len(video.video_log_paths) < self.num_frames_in_video:
                continue

            trimmed_log = OpenWorldVideoLogSingleton(
                video_log_paths=video.video_log_paths[: self.num_frames_in_video],
                video_id=video.video_id,
            )
            valid_videos.append(trimmed_log)

        valid_videos = list(set(valid_videos))[:limit]
        return OpenWorldVideoLog(video_logs=valid_videos)

    def setup(self, train_percentage: float = 0.9):
        # Now ensure we are limiting to the limit (num_frames_in_video, limit)
        train_dataset, test_dataset = self._get_or_create_dataset(train_percentage)

        # Filter to valid videos
        train_dataset = self._filter_to_valid_videos(train_dataset, self.limit)
        test_dataset = self._filter_to_valid_videos(test_dataset, self.limit)

        return train_dataset, test_dataset

    def _get_dataset_key(self, stage: Literal["train", "test"], split: float) -> str:
        formatted_split = str(split).replace(".", "_")
        return f"{self.dataset_dir}_{stage}_{formatted_split}_{self.num_frames_in_video}_frames.json"

    def _load_existing_dataset(
        self,
        key: str,
    ) -> OpenWorldVideoLog:
        with open(key, "r") as f:
            return OpenWorldVideoLog.model_validate_json(f.read())

    def _save_dataset(self, dataset: OpenWorldVideoLog, key: str):
        with open(key, "w") as f:
            f.write(dataset.model_dump_json())

    def _load_image_locally(self, frame_path: str) -> Image.Image | None:
        return self.local_cache.get(frame_path)

    @staticmethod
    def _load_image_from_s3(
        s3_manager: S3Manager, local_cache: Cache, s3_key: str
    ) -> Image.Image | None:
        """Load image from S3 with local caching"""
        try:
            image = local_cache.get(s3_key)
            if image is not None:
                return image.convert("RGB")

            # Download from S3
            image = s3_manager.download_image(s3_key)
            if image is None:
                logger.error(f"Failed to download image from S3: {s3_key}")
                return None

            # Cache the image locally
            local_cache.set(s3_key, image)
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image from S3: {e}")
            return None

    @staticmethod
    def _get_chunked_frames_with_paths(
        loaded_images: list[Image.Image],
        frame_paths: list[str],
        batch_size: int,
        overlap: int,
    ) -> list[list[FrameWithPath]]:
        chunked_frames_with_paths: list[list[FrameWithPath]] = []

        for j in range(0, len(loaded_images), batch_size):
            chunked_frames_with_paths.append(
                [
                    FrameWithPath(path=path, frame=image)
                    for image, path in zip(
                        loaded_images[j : j + batch_size + overlap],
                        frame_paths[j : j + batch_size + overlap],
                    )
                ]
            )

        return chunked_frames_with_paths

    @staticmethod
    def _get_frame_sequences(
        chunks: list[S3Frame],
        limit: int,
        max_frame_sequence_length: int,
        num_frames_in_video: int,
        local_cache: Cache,
    ) -> list[OpenWorldVideoLogSingleton]:
        videos: list[OpenWorldVideoLogSingleton] = []

        def get_valid_frame_sequence(
            frame_list: list[FrameWithPath], progress_bar: tqdm
        ):
            if len(videos) >= limit:
                progress_bar.update(1)
                return

            if len(frame_list) > max_frame_sequence_length:
                progress_bar.update(1)
                raise ValueError(
                    f"Frame sequence length is greater than {max_frame_sequence_length}"
                )

            if len(frame_list) < num_frames_in_video:
                progress_bar.update(1)
                return

            frames = filter_frame_sequence(frame_list)
            if len(frames) >= num_frames_in_video:
                videos.append(
                    OpenWorldVideoLogSingleton(
                        video_log_paths=frames, video_id=frame_list[0].path
                    )
                )
                logger.info(f"Added video with {len(frames)} frames")
            progress_bar.update(1)

        with ThreadPoolExecutor(max_workers=10) as thread_executor:
            frame_paths = [path.obj_key for path in chunks]
            partial_fn = partial(
                OpenWorldRunningDatasetCreator._load_image_from_s3,
                default_s3_manager,
                local_cache,
            )
            loaded_images = list(
                thread_executor.map(
                    partial_fn,
                    frame_paths,
                )
            )
            loaded_images = [image for image in loaded_images if image is not None]

            overlap = num_frames_in_video
            frames_with_paths = (
                OpenWorldRunningDatasetCreator._get_chunked_frames_with_paths(
                    loaded_images, frame_paths, num_frames_in_video, overlap
                )
            )
            progress_bar = tqdm(
                total=len(frames_with_paths),
                desc=f"Processing {len(frames_with_paths)} frames with paths",
            )

            partial_fn = partial(get_valid_frame_sequence, progress_bar=progress_bar)

            thread_executor.map(partial_fn, frames_with_paths)

        return videos

    def _pull_dataset_from_s3(
        self, source_dir: str
    ) -> list[OpenWorldVideoLogSingleton]:
        directories_to_process = list_frames_in_s3(default_s3_manager, source_dir)

        all_videos: list[OpenWorldVideoLogSingleton] = []
        for directory in tqdm(
            directories_to_process,
            desc=f"Processing {len(directories_to_process)} directories",
        ):
            directory.frame_list.sort(key=lambda x: x.frame_num)
            directory.frame_list = directory.frame_list

            # Work over chunks of at most self.frame_chunk_size frames at once
            chunks = [
                directory.frame_list[i : i + self.frame_chunk_size]
                for i in range(0, len(directory.frame_list), self.frame_chunk_size)
            ]

            with Pool(10) as process_executor:
                partial_fn = partial(
                    self._get_frame_sequences,
                    limit=self.limit,
                    max_frame_sequence_length=self.max_frame_sequence_length,
                    num_frames_in_video=self.num_frames_in_video,
                    local_cache=self.local_cache,
                )
                videos = process_executor.map(partial_fn, chunks)
                logger.info(f"Found {len(videos)} videos")

                all_videos.extend(sum(videos, []))

        return all_videos
