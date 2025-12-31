#!/usr/bin/env python3
"""
Pokemon Frame Extractor

Converts clean Pokemon videos to PNG frames for dataset creation.
Extracts frames at 5fps and 360p resolution, removing audio and applying crops.
Supports both local storage and S3.

Performance optimizations:
- Frame jumping: skips invalid frames (non-navigable environments) in 5-second chunks
- Multiprocessing: processes multiple videos in parallel
- Multithreading: S3 uploads are performed asynchronously
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

from data.s3.s3_upload_worker import S3UploadWorker
from data.s3.s3_utils import S3Manager, default_s3_manager
from data.scraping.frame_metadata import FrameMetadata
from data.scraping.video_cleaner import CropRegion, PokemonVideoCleaner

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


GAME_MAPPINGS = {
    "emerald": "Pokemon Emerald",
    "fire_red": "Pokemon Fire Red",
    "firered": "Pokemon Fire Red",
    "ruby": "Pokemon Ruby",
    "sapphire": "Pokemon Sapphire",
    "heart_gold": "Pokemon Heart Gold",
    "heartgold": "Pokemon Heart Gold",
    "soul_silver": "Pokemon Soul Silver",
    "soulsilver": "Pokemon Soul Silver",
}


class PokemonFrameExtractor:
    """Extracts frames from clean Pokemon videos with S3 support."""

    def __init__(
        self,
        target_fps: int = 5,
        target_height: int = 360,
        use_s3: bool = False,
        s3_manager: Optional[S3Manager] = None,
        raw_s3_prefix: str = "raw_videos",
        frames_s3_prefix: str = "pokemon_frames",
        jump_seconds: float = 5.0,
        num_video_workers: int = 4,
        num_upload_threads: int = 8,
    ):
        """
        Initialize the Pokemon frame extractor.

        Args:
            target_fps: Target frames per second for extraction
            target_height: Target height for extracted frames
            use_s3: Whether to use S3 for storage
            s3_manager: S3Manager instance (if None and use_s3=True, will create from env)
            raw_s3_prefix: S3 prefix for raw videos
            frames_s3_prefix: S3 prefix for extracted frames
            jump_seconds: Seconds to skip when encountering invalid frames
            num_video_workers: Number of parallel video processing workers
            num_upload_threads: Number of threads for S3 uploads
        """
        self.target_fps = target_fps
        self.target_height = target_height
        self.use_s3 = use_s3
        self.raw_s3_prefix = raw_s3_prefix
        self.frames_s3_prefix = frames_s3_prefix
        self.jump_seconds = jump_seconds
        self.num_video_workers = num_video_workers
        self.num_upload_threads = num_upload_threads

        self.cleaner = PokemonVideoCleaner(use_s3=use_s3)
        self._s3_worker: Optional[S3UploadWorker] = None

    # -------------------------------------------------------------------------
    # S3 Operations
    # -------------------------------------------------------------------------

    def _get_s3_worker(self) -> S3UploadWorker:
        """Get or create the S3 upload worker for this process."""
        if self._s3_worker is None:
            self._s3_worker = S3UploadWorker(max_workers=self.num_upload_threads)
            self._s3_worker.start()
        return self._s3_worker

    def _download_video_from_s3(self, s3_key: str) -> Optional[str]:
        """Download video from S3 to temporary file."""
        if not self.use_s3:
            return None

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            success = default_s3_manager.download_file(s3_key, temp_path)
            if success:
                logger.debug(f"Downloaded video from S3: {s3_key}")
                return temp_path

            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

        except Exception as e:
            logger.error(f"Error downloading video from S3: {e}")
            return None

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file."""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {temp_path}: {e}")

    def _upload_frame_to_s3(
        self,
        local_path: str,
        game_name: str,
        video_id: str,
        frame_filename: str,
        delete_after_upload: bool = False,
    ):
        """Queue a frame for async upload to S3."""
        if not self.use_s3:
            return

        game_dir = game_name.replace(" ", "_").lower()
        s3_key = f"{self.frames_s3_prefix}/{game_dir}/{video_id}/{frame_filename}"

        worker = self._get_s3_worker()
        worker.submit_frame_upload(local_path, s3_key, delete_after_upload)

    def _upload_metadata_to_s3(
        self,
        metadata_dict: Dict[str, Any],
        game_name: str,
        video_id: str,
        filename: str,
    ):
        """Queue metadata for async upload to S3."""
        if not self.use_s3:
            return

        game_dir = game_name.replace(" ", "_").lower()
        s3_key = f"{self.frames_s3_prefix}/{game_dir}/{video_id}/{filename}"

        worker = self._get_s3_worker()
        worker.submit_metadata_upload(metadata_dict, s3_key)

    # -------------------------------------------------------------------------
    # Frame Extraction Core
    # -------------------------------------------------------------------------

    def calculate_target_size(
        self, crop_width: int, crop_height: int
    ) -> Tuple[int, int]:
        """Calculate target size maintaining aspect ratio."""
        aspect_ratio = crop_width / crop_height
        target_width = int(self.target_height * aspect_ratio)

        # Ensure even dimensions for video compatibility
        target_width = target_width + (target_width % 2)
        target_height = self.target_height + (self.target_height % 2)

        return target_width, target_height

    def _is_valid_frame(self, pil_image: Image.Image) -> bool:
        """Check if a frame is valid (navigable environment)."""
        # return is_frame_within_navigable_environment(pil_image)
        return True

    def extract_frames_from_video(
        self,
        video_path: str,
        crop_region: CropRegion,
        output_dir: str,
        game_name: str,
        keep_local_frames: bool = True,
    ) -> List[str]:
        """
        Extract frames from a video with cropping, resizing, and frame jumping.

        Uses intelligent frame jumping: when an invalid frame is detected,
        skips ahead by jump_seconds to quickly pass through non-gameplay sections.
        """
        temp_file = None
        actual_video_path = video_path

        # Check if this is an S3 path and download if needed
        if self.use_s3 and (
            video_path.startswith("s3://") or not os.path.exists(video_path)
        ):
            s3_key = (
                video_path
                if not video_path.startswith("s3://")
                else video_path[5:].split("/", 1)[1]
            )
            temp_file = self._download_video_from_s3(s3_key)
            if not temp_file:
                logger.error(f"Could not download video from S3: {s3_key}")
                return []
            actual_video_path = temp_file

        try:
            cap = cv2.VideoCapture(actual_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {actual_video_path}")
                return []

            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / original_fps if original_fps > 0 else 0

            logger.info(f"Processing video: {video_path}")
            logger.info(
                f"Original: {original_width}x{original_height}, {original_fps:.1f}fps, {duration:.1f}s"
            )

            # Calculate intervals
            frame_interval = max(1, int(original_fps / self.target_fps))
            jump_frames = int(original_fps * self.jump_seconds)

            # Skip first and last 3 minutes of video
            start_frame = int(original_fps * 180)
            end_frame = total_frames - int(original_fps * 180)

            # Calculate target size
            target_width, target_height = self.calculate_target_size(
                crop_region.width, crop_region.height
            )

            # Setup output directory
            video_id = Path(video_path).stem
            video_output_dir = (
                Path(output_dir) / game_name.replace(" ", "_").lower() / video_id
            )
            video_output_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames with progress bar
            extracted_frames = self._process_video_frames(
                cap=cap,
                crop_region=crop_region,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_interval=frame_interval,
                jump_frames=jump_frames,
                target_width=target_width,
                target_height=target_height,
                original_fps=original_fps,
                original_width=original_width,
                original_height=original_height,
                video_id=video_id,
                video_output_dir=video_output_dir,
                game_name=game_name,
                keep_local_frames=keep_local_frames,
            )

            cap.release()

            # Wait for S3 uploads to complete
            if self.use_s3 and self._s3_worker:
                success, failed = self._s3_worker.wait_for_completion()
                logger.info(
                    f"S3 uploads for {video_id}: {success} successful, {failed} failed"
                )

            # Save video summary
            self._save_video_summary(
                video_path=video_path,
                crop_region=crop_region,
                game_name=game_name,
                frame_count=len(extracted_frames),
                output_dir=video_output_dir,
                keep_local=keep_local_frames,
            )

            logger.info(f"Extracted {len(extracted_frames)} frames from {video_id}")
            return extracted_frames

        finally:
            if temp_file:
                self._cleanup_temp_file(temp_file)

    def _process_video_frames(
        self,
        cap: cv2.VideoCapture,
        crop_region: CropRegion,
        start_frame: int,
        end_frame: int,
        frame_interval: int,
        jump_frames: int,
        target_width: int,
        target_height: int,
        original_fps: float,
        original_width: int,
        original_height: int,
        video_id: str,
        video_output_dir: Path,
        game_name: str,
        keep_local_frames: bool,
    ) -> List[str]:
        """Process frames from video with progress tracking."""
        extracted_frames: List[str] = []
        frame_count = 0
        frame_idx = start_frame
        x, y, w, h = crop_region.x, crop_region.y, crop_region.width, crop_region.height

        # Progress bar based on frame position
        total_frames_to_process = end_frame - start_frame
        pbar = tqdm(
            total=total_frames_to_process, desc=f"Extracting {video_id}", unit="frames"
        )

        prev_valid_frame_path = None
        valid_frame_pbar = tqdm(
            total=total_frames_to_process,
            desc=f"Valid frames for {video_id}",
            unit="frames",
        )
        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                frame_idx += frame_interval
                pbar.update(frame_interval)
                prev_valid_frame_path = None
                continue

            # Apply crop
            cropped_frame = frame[y : y + h, x : x + w]
            if cropped_frame.size == 0:
                frame_idx += frame_interval
                pbar.update(frame_interval)
                prev_valid_frame_path = None
                continue

            # Resize to target resolution
            resized_frame = cv2.resize(
                cropped_frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Check if frame is valid (navigable environment)
            if not self._is_valid_frame(pil_image):
                # If the previous frame was valid, this might just be a brief UI
                # interruption (text banner, etc). Use a small step to not skip
                # past the resumption of valid gameplay.
                if prev_valid_frame_path is not None:
                    # Previous was valid - use small step to find when gameplay resumes
                    frame_idx += frame_interval
                    pbar.update(frame_interval)
                else:
                    # We're in a long invalid section - use full jump
                    frame_idx += jump_frames
                    pbar.update(jump_frames)
                prev_valid_frame_path = None
                continue

            # Save frame
            timestamp = frame_idx / original_fps
            frame_filename = f"frame_{frame_idx:06d}_{timestamp:.2f}s.png"
            frame_path = video_output_dir / frame_filename

            pil_image.save(frame_path, "PNG", optimize=True)

            # Queue S3 upload
            if self.use_s3:
                self._upload_frame_to_s3(
                    str(frame_path),
                    game_name,
                    video_id,
                    frame_filename,
                    delete_after_upload=not keep_local_frames,
                )

            # Save metadata
            metadata = FrameMetadata(
                video_id=video_id,
                frame_number=frame_idx,
                timestamp=timestamp,
                game=game_name,
                original_resolution=(original_width, original_height),
                cropped_resolution=(crop_region.width, crop_region.height),
                final_resolution=(target_width, target_height),
                prev_frame_key=prev_valid_frame_path,
            )
            self._save_frame_metadata(
                metadata, video_output_dir, frame_filename, game_name, video_id
            )

            extracted_frames.append(str(frame_path))
            frame_count += 1
            frame_idx += frame_interval
            pbar.update(frame_interval)
            prev_valid_frame_path = frame_path.name
            valid_frame_pbar.update(1)

        pbar.close()
        valid_frame_pbar.close()
        return extracted_frames

    def _save_frame_metadata(
        self,
        metadata: FrameMetadata,
        video_output_dir: Path,
        image_filename: str,
        game_name: str,
        video_id: str,
    ):
        """Save frame metadata to JSON and optionally upload to S3."""
        metadata_dict = metadata.__dict__
        metadata_filename = image_filename.replace(".png", ".json")
        metadata_path = video_output_dir / metadata_filename

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        if self.use_s3:
            self._upload_metadata_to_s3(
                metadata_dict, game_name, video_id, metadata_filename
            )

    def _save_video_summary(
        self,
        video_path: str,
        crop_region: CropRegion,
        game_name: str,
        frame_count: int,
        output_dir: Path,
        keep_local: bool = True,
    ):
        """Save summary information for the processed video."""
        summary = {
            "source_video": video_path,
            "game": game_name,
            "crop_region": {
                "x": crop_region.x,
                "y": crop_region.y,
                "width": crop_region.width,
                "height": crop_region.height,
                "confidence": crop_region.confidence,
            },
            "extraction_settings": {
                "target_fps": self.target_fps,
                "target_height": self.target_height,
                "jump_seconds": self.jump_seconds,
            },
            "results": {
                "frames_extracted": frame_count,
                "output_directory": str(output_dir),
            },
            "storage": {
                "use_s3": self.use_s3,
                "keep_local_frames": keep_local,
            },
        }

        summary_path = output_dir / "video_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_s3:
            video_id = Path(video_path).stem
            self._upload_metadata_to_s3(
                summary, game_name, video_id, "video_summary.json"
            )

    # -------------------------------------------------------------------------
    # Video Processing
    # -------------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        game_name: str,
        output_dir: str,
        keep_local_frames: bool = True,
    ) -> bool:
        """Process a single video: clean, crop, and extract frames."""
        logger.info(f"Processing video: {video_path}")

        is_clean, crop_region = self.cleaner.is_video_clean(video_path)

        if not is_clean or not crop_region:
            logger.warning(f"Video failed cleaning validation: {video_path}")
            return False

        try:
            extracted_frames = self.extract_frames_from_video(
                video_path, crop_region, output_dir, game_name, keep_local_frames
            )

            if extracted_frames:
                logger.info(
                    f"Successfully processed video: {video_path} ({len(extracted_frames)} frames)"
                )
                return True

            logger.warning(f"No frames extracted from video: {video_path}")
            return False

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage configuration."""
        return {
            "use_s3": self.use_s3,
            "s3_bucket": default_s3_manager.bucket_name,
            "raw_s3_prefix": self.raw_s3_prefix,
            "frames_s3_prefix": self.frames_s3_prefix,
            "target_fps": self.target_fps,
            "target_height": self.target_height,
            "jump_seconds": self.jump_seconds,
            "num_video_workers": self.num_video_workers,
            "num_upload_threads": self.num_upload_threads,
        }

    # -------------------------------------------------------------------------
    # Directory Processing
    # -------------------------------------------------------------------------

    def process_video_directory(
        self,
        video_dir: str,
        output_dir: str,
        keep_local_frames: bool = True,
        use_multiprocessing: bool = True,
    ):
        """
        Process all videos in a directory.

        Args:
            video_dir: Directory containing video files
            output_dir: Output directory for extracted frames
            keep_local_frames: Whether to keep local frame copies when using S3
            use_multiprocessing: Whether to use multiprocessing (default True)
        """
        video_path = Path(video_dir)

        if not video_path.exists():
            logger.error(f"Video directory does not exist: {video_dir}")
            return

        video_files = self._find_video_files(video_path)

        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return

        logger.info(f"Found {len(video_files)} video files to process")

        if use_multiprocessing and len(video_files) > 1:
            self._process_videos_parallel(video_files, output_dir, keep_local_frames)
        else:
            self._process_videos_sequential(video_files, output_dir, keep_local_frames)

    def _find_video_files(self, video_path: Path) -> List[Path]:
        """Find all video files in directory."""
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(video_path.glob(f"**/*{ext}"))

        return video_files

    def _process_videos_sequential(
        self,
        video_files: List[Path],
        output_dir: str,
        keep_local_frames: bool,
    ):
        """Process videos sequentially."""
        processed_count = 0
        failed_count = 0

        for video_file in tqdm(video_files, desc="Processing videos"):
            game_name = self._determine_game_name(video_file)

            if self.process_video(
                str(video_file), game_name, output_dir, keep_local_frames
            ):
                processed_count += 1
            else:
                failed_count += 1
                self._handle_failed_video(video_file)

        if self._s3_worker:
            self._s3_worker.stop()
            self._s3_worker = None

        logger.info(
            f"Processing complete: {processed_count} successful, {failed_count} failed"
        )

    def _process_videos_parallel(
        self,
        video_files: List[Path],
        output_dir: str,
        keep_local_frames: bool,
    ):
        """Process videos in parallel using multiprocessing."""
        video_args = [
            (
                str(video_file),
                self._determine_game_name(video_file),
                output_dir,
                keep_local_frames,
                self.target_fps,
                self.target_height,
                self.use_s3,
                self.raw_s3_prefix,
                self.frames_s3_prefix,
                self.jump_seconds,
                self.num_upload_threads,
            )
            for video_file in video_files
        ]

        num_workers = min(self.num_video_workers, len(video_files))
        logger.info(f"Processing {len(video_files)} videos with {num_workers} workers")

        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_single_video_worker, video_args),
                    total=len(video_args),
                    desc="Processing videos",
                )
            )

        processed_count = sum(1 for r in results if r)
        failed_count = sum(1 for r in results if not r)

        for video_file, success in zip(video_files, results, strict=False):
            if not success:
                self._handle_failed_video(video_file)

        logger.info(
            f"Processing complete: {processed_count} successful, {failed_count} failed"
        )

    def _determine_game_name(self, video_path: Path) -> str:
        """Determine Pokemon game name from video path."""
        path_str = str(video_path).lower()

        for key, game_name in GAME_MAPPINGS.items():
            if key in path_str:
                return game_name

        return "Pokemon Unknown"

    def _handle_failed_video(self, video_path: Path):
        """Handle videos that failed processing."""
        failed_dir = video_path.parent / "failed_videos"
        failed_dir.mkdir(exist_ok=True)

        try:
            new_path = failed_dir / video_path.name
            video_path.rename(new_path)
            logger.info(f"Moved failed video to: {new_path}")
        except Exception as e:
            logger.error(f"Could not move failed video {video_path}: {e}")

    # -------------------------------------------------------------------------
    # Dataset Index
    # -------------------------------------------------------------------------

    def create_dataset_index(self, output_dir: str) -> Dict[str, Any]:
        """Create an index of all extracted frames for easy dataset loading."""
        output_path = Path(output_dir)

        dataset_index: Dict[str, Any] = {
            "games": {},
            "total_frames": 0,
            "extraction_settings": {
                "target_fps": self.target_fps,
                "target_height": self.target_height,
                "jump_seconds": self.jump_seconds,
            },
        }

        for game_dir in output_path.iterdir():
            if not game_dir.is_dir():
                continue

            game_name = game_dir.name
            dataset_index["games"][game_name] = {"videos": {}, "frame_count": 0}

            for video_dir in game_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                video_id = video_dir.name
                frame_files = list(video_dir.glob("frame_*.png"))
                frame_count = len(frame_files)

                if frame_count > 0:
                    dataset_index["games"][game_name]["videos"][video_id] = {
                        "frame_count": frame_count,
                        "directory": str(video_dir),
                    }
                    dataset_index["games"][game_name]["frame_count"] += frame_count
                    dataset_index["total_frames"] += frame_count

        index_path = output_path / "dataset_index.json"
        with open(index_path, "w") as f:
            json.dump(dataset_index, f, indent=2)

        logger.info(f"Created dataset index: {index_path}")
        logger.info(f"Total frames in dataset: {dataset_index['total_frames']}")

        return dataset_index


# =============================================================================
# Multiprocessing Worker
# =============================================================================


def _process_single_video_worker(args: tuple) -> bool:
    """
    Worker function for multiprocessing video extraction.

    This is a module-level function to allow pickling for multiprocessing.
    Creates a new PokemonFrameExtractor instance per process.
    """
    (
        video_path,
        game_name,
        output_dir,
        keep_local_frames,
        target_fps,
        target_height,
        use_s3,
        raw_s3_prefix,
        frames_s3_prefix,
        jump_seconds,
        num_upload_threads,
    ) = args

    try:
        extractor = PokemonFrameExtractor(
            target_fps=target_fps,
            target_height=target_height,
            use_s3=use_s3,
            raw_s3_prefix=raw_s3_prefix,
            frames_s3_prefix=frames_s3_prefix,
            jump_seconds=jump_seconds,
            num_upload_threads=num_upload_threads,
        )

        result = extractor.process_video(
            video_path, game_name, output_dir, keep_local_frames
        )

        if extractor._s3_worker:
            extractor._s3_worker.stop()

        return result

    except Exception as e:
        logger.error(f"Worker error processing {video_path}: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main function for testing."""
    extractor = PokemonFrameExtractor(
        target_fps=5,
        target_height=360,
        jump_seconds=5.0,
        num_video_workers=4,
        num_upload_threads=8,
    )

    input_dir = "raw_videos"
    output_dir = "pokemon_frames"

    if os.path.exists(input_dir):
        extractor.process_video_directory(input_dir, output_dir)
        extractor.create_dataset_index(output_dir)
    else:
        logger.warning(f"Input directory {input_dir} does not exist")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
