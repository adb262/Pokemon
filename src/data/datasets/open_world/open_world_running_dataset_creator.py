import json
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
from data.s3.gather_frames_in_dirs import S3Frame, list_frames_in_s3
from data.s3.s3_utils import S3Manager, default_s3_manager
from data.scraping.frame_extractor import FrameMetadata
from data.scraping.frame_filterer import FrameWithPath, filter_frame_sequence

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
        use_s3: bool,
    ):
        self.dataset_dir = dataset_dir
        self.num_frames_in_video = num_frames_in_video
        self.output_log_dir = output_log_json_file_name
        self.local_cache = local_cache
        self.limit = limit
        self.image_size = image_size
        self.use_s3 = use_s3

    def _get_or_create_dataset(
        self, train_percentage: float = 0.9
    ) -> tuple[OpenWorldVideoLog, OpenWorldVideoLog]:
        train_dataset_key = self._get_dataset_key("train", train_percentage)
        test_dataset_key = self._get_dataset_key("test", train_percentage)
        logger.info(f"Train dataset key: {train_dataset_key}")
        logger.info(f"Test dataset key: {test_dataset_key}")
        if os.path.exists(train_dataset_key) and os.path.exists(test_dataset_key):
            return (
                self.load_existing_dataset(train_dataset_key),
                self.load_existing_dataset(test_dataset_key),
            )

        data = self._pull_dataset(self.dataset_dir)
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

    @staticmethod
    def load_existing_dataset(
        key: str,
    ) -> OpenWorldVideoLog:
        with open(key, "r") as f:
            return OpenWorldVideoLog.model_validate_json(f.read())

    def _save_dataset(self, dataset: OpenWorldVideoLog, key: str):
        with open(key, "w") as f:
            f.write(dataset.model_dump_json())

    def _load_image_locally(self, frame_path: str) -> Image.Image | None:
        return self.local_cache.get(frame_path)

    def _load_metadata_locally(self, frame_path: str) -> FrameMetadata | None:
        return FrameMetadata(**json.load(open(frame_path.replace(".png", ".json"))))

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
    def _load_metadata_from_s3(
        s3_manager: S3Manager, s3_key: str
    ) -> FrameMetadata | None:
        try:
            metadata = s3_manager.download_json(s3_key)
            if metadata is not None:
                return FrameMetadata(**metadata)

            return None
        except Exception as e:
            logger.error(f"Error loading metadata from S3: {e}")
            return None

    @staticmethod
    def _get_chunked_frames_with_paths(
        frames_with_paths: list[FrameWithPath],
    ) -> list[list[FrameWithPath]]:
        chunked_frames_with_paths: list[list[FrameWithPath]] = []

        current_chunk = [frames_with_paths[0]]
        for frame in frames_with_paths[1:]:
            if current_chunk[-1].path.split("/")[-1] != frame.metadata.prev_frame_key:
                logger.info(
                    f"Adding chunk of {len(current_chunk)} frames. index: {current_chunk[-1].path}. frame_idx: {frame.path}. "
                )
                chunked_frames_with_paths.append(current_chunk)
                current_chunk = [frame]
            else:
                current_chunk.append(frame)
        chunked_frames_with_paths.append(current_chunk)

        return chunked_frames_with_paths

    @staticmethod
    def _get_valid_frame_sequences(
        *,
        frame_list: list[FrameWithPath],
        progress_bar: tqdm,
        limit: int,
        num_frames_in_video: int,
    ) -> list[OpenWorldVideoLogSingleton]:
        if progress_bar.n >= limit:
            return []

        if len(frame_list) < num_frames_in_video:
            return []

        frames = filter_frame_sequence(frame_list, num_frames_in_video)
        videos: list[OpenWorldVideoLogSingleton] = [
            OpenWorldVideoLogSingleton(
                video_log_paths=frame_sequence, video_id=frame_list[0].metadata.video_id
            )
            for frame_sequence in frames
        ]
        logger.info(f"Added {len(videos)} videos")
        progress_bar.update(len(videos))
        return videos

    def _get_frame_sequences_from_local(
        self,
        frame_paths: list[str],
        limit: int,
        num_frames_in_video: int,
        progress_bar: tqdm,
    ) -> list[OpenWorldVideoLogSingleton]:
        frame_with_paths: list[FrameWithPath] = []
        metadata = [self._load_metadata_locally(path) for path in frame_paths]
        images = [self._load_image_locally(path) for path in frame_paths]
        for path, image, metadata in zip(frame_paths, images, metadata):
            if image is not None and metadata is not None:
                frame_with_paths.append(
                    FrameWithPath(path=path, frame=image, metadata=metadata)
                )

        if not frame_with_paths:
            return []

        frames_with_paths = (
            OpenWorldRunningDatasetCreator._get_chunked_frames_with_paths(
                frame_with_paths,
            )
        )

        all_videos = [
            OpenWorldRunningDatasetCreator._get_valid_frame_sequences(
                progress_bar=progress_bar,
                limit=limit,
                num_frames_in_video=num_frames_in_video,
                frame_list=frames_with_paths,
            )
            for frames_with_paths in frames_with_paths
        ]

        return sum(all_videos, [])

    def _get_frame_sequences_from_s3(
        self,
        chunks: list[S3Frame],
        limit: int,
        num_frames_in_video: int,
        local_cache: Cache,
    ) -> list[OpenWorldVideoLogSingleton]:
        with ThreadPoolExecutor(max_workers=10) as thread_executor:
            frame_paths = [path.obj_key for path in chunks]
            metadata_paths = [path.metadata_obj_key for path in chunks]
            partial_fn = partial(
                OpenWorldRunningDatasetCreator._load_image_from_s3,
                default_s3_manager,
                local_cache,
            )
            partial_metadata_fn = partial(
                OpenWorldRunningDatasetCreator._load_metadata_from_s3,
                default_s3_manager,
            )
            loaded_images = list(
                thread_executor.map(
                    partial_fn,
                    frame_paths,
                )
            )
            loaded_metadata = list[FrameMetadata | None](
                thread_executor.map(
                    partial_metadata_fn,
                    metadata_paths,
                )
            )
            frame_with_paths: list[FrameWithPath] = []
            for path, image, metadata in zip(
                frame_paths, loaded_images, loaded_metadata
            ):
                if image is not None and metadata is not None:
                    frame_with_paths.append(
                        FrameWithPath(path=path, frame=image, metadata=metadata)
                    )

            frames_with_paths = (
                OpenWorldRunningDatasetCreator._get_chunked_frames_with_paths(
                    frame_with_paths,
                )
            )
            progress_bar = tqdm(
                total=len(frames_with_paths),
                desc=f"Processing {len(frames_with_paths)} frames with paths",
            )

            partial_fn = partial(
                OpenWorldRunningDatasetCreator._get_valid_frame_sequences,
                progress_bar=progress_bar,
                limit=limit,
                num_frames_in_video=num_frames_in_video,
            )

            videos = list(thread_executor.map(partial_fn, frames_with_paths))

        return sum(videos, [])

    def _pull_dataset(self, source_dir: str) -> list[OpenWorldVideoLogSingleton]:
        if self.use_s3:
            return self._pull_dataset_from_s3(source_dir)
        else:
            return self._pull_dataset_from_local(source_dir)

    @staticmethod
    def _unpack_directory(directory: str) -> list[str]:
        # Frame files are in the format of frame_<frame_number>.png
        if not os.path.isdir(directory):
            return []

        items = os.listdir(directory)
        frame_items = [
            os.path.join(directory, item) for item in items if item.endswith(".png")
        ]
        subdirectories = [
            os.path.join(directory, item)
            for item in items
            if os.path.isdir(os.path.join(directory, item))
        ]
        for subdirectory in subdirectories:
            frame_items.extend(
                OpenWorldRunningDatasetCreator._unpack_directory(subdirectory)
            )

        return frame_items

    def _pull_dataset_from_local(
        self, source_dir: str
    ) -> list[OpenWorldVideoLogSingleton]:
        directories_to_process = os.listdir(source_dir)

        all_videos: list[OpenWorldVideoLogSingleton] = []
        progress_bar = tqdm(
            total=self.limit,
            desc=f"Processing {self.limit} frames with paths",
        )
        for directory in tqdm(
            directories_to_process,
            desc=f"Processing {len(directories_to_process)} directories",
        ):
            frames = sorted(
                OpenWorldRunningDatasetCreator._unpack_directory(
                    os.path.join(source_dir, directory)
                )
            )
            chunks: list[list[str]] = [
                frames[i : i + self.frame_chunk_size]
                for i in range(0, len(frames), self.frame_chunk_size)
            ]

            # with Pool(2) as process_executor:
            #     partial_fn = partial(
            #         self._get_frame_sequences_from_local,
            #         limit=self.limit,
            #         num_frames_in_video=self.num_frames_in_video,
            #         progress_bar=progress_bar,
            #     )
            #     videos = process_executor.map(partial_fn, chunks)
            #     logger.info(f"Found {len(videos)} videos")

            #     all_videos.extend(sum(videos, []))

            for frame_path in chunks:
                all_videos.extend(
                    self._get_frame_sequences_from_local(
                        frame_paths=frame_path,
                        limit=self.limit,
                        num_frames_in_video=self.num_frames_in_video,
                        progress_bar=progress_bar,
                    )
                )
                if len(all_videos) >= self.limit:
                    break

        return all_videos

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
                    self._get_frame_sequences_from_s3,
                    limit=self.limit,
                    num_frames_in_video=self.num_frames_in_video,
                    local_cache=self.local_cache,
                )
                videos = process_executor.map(partial_fn, chunks)
                logger.info(f"Found {len(videos)} videos")

                all_videos.extend(sum(videos, []))

        return all_videos

    def show_sample_grid(
        self,
        dataset: OpenWorldVideoLog,
        num_trajectories: int = 20,
        max_frames_per_trajectory: int = 10,
        frame_size: tuple[int, int] = (128, 128),
        padding: int = 4,
    ):
        # Sample trajectories from the dataset
        num_to_sample = min(num_trajectories, len(dataset))
        sampled_trajectories = random.sample(dataset.video_logs, num_to_sample)

        # Determine grid dimensions
        num_cols = min(
            max_frames_per_trajectory,
            max(len(t.video_log_paths) for t in sampled_trajectories),
        )
        num_rows = len(sampled_trajectories)

        # Calculate final image dimensions
        cell_width = frame_size[0] + padding
        cell_height = frame_size[1] + padding
        grid_width = num_cols * cell_width + padding
        grid_height = num_rows * cell_height + padding

        # Create the grid image with a dark background
        grid_image = Image.new("RGB", (grid_width, grid_height), color=(30, 30, 30))

        logger.info(
            f"Creating sample grid with {num_rows} trajectories, {num_cols} frames each"
        )

        # Load and place frames
        for row_idx, trajectory in enumerate(
            tqdm(sampled_trajectories, desc="Loading trajectories")
        ):
            frame_paths = trajectory.video_log_paths[:max_frames_per_trajectory]

            for col_idx, frame_path in enumerate(frame_paths):
                # Try to load from cache first, then from S3
                image = self._load_image_locally(frame_path)
                if image is None:
                    image = self._load_image_from_s3(
                        default_s3_manager, self.local_cache, frame_path
                    )

                if image is None:
                    # Create a placeholder for missing images
                    image = Image.new("RGB", frame_size, color=(80, 80, 80))
                else:
                    # Resize the image to the target frame size
                    image = image.resize(frame_size, Image.Resampling.LANCZOS)

                # Calculate position and paste
                x = padding + col_idx * cell_width
                y = padding + row_idx * cell_height
                grid_image.paste(image, (x, y))

        # Save the grid
        grid_image.show()
