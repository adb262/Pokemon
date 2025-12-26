#!/usr/bin/env python3
"""
Pokemon Frame Extractor
Converts clean Pokemon videos to PNG frames for dataset creation.
Extracts frames at 5fps and 360p resolution, removing audio and applying crops.
Supports both local storage and S3.
"""

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

from data.scraping.video_cleaner import CropRegion, PokemonVideoCleaner
from s3.s3_utils import S3Manager, default_s3_manager

# Add the parent directory to the path so we can import from idm
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for extracted frames"""

    video_id: str
    frame_number: int
    timestamp: float
    game: str
    original_resolution: Tuple[int, int]
    cropped_resolution: Tuple[int, int]
    final_resolution: Tuple[int, int]


class PokemonFrameExtractor:
    """Extracts frames from clean Pokemon videos with S3 support"""

    def __init__(
        self,
        target_fps: int = 5,
        target_height: int = 360,
        use_s3: bool = False,
        s3_manager: Optional[S3Manager] = None,
        raw_s3_prefix: str = "raw_videos",
        frames_s3_prefix: str = "pokemon_frames",
    ):
        """
        Initialize the Pokemon frame extractor

        Args:
            target_fps: Target frames per second for extraction
            target_height: Target height for extracted frames
            use_s3: Whether to use S3 for storage
            s3_manager: S3Manager instance (if None and use_s3=True, will create from env)
            raw_s3_prefix: S3 prefix for raw videos
            frames_s3_prefix: S3 prefix for extracted frames
        """
        self.target_fps = target_fps
        self.target_height = target_height
        self.use_s3 = use_s3
        self.raw_s3_prefix = raw_s3_prefix
        self.frames_s3_prefix = frames_s3_prefix

        self.cleaner = PokemonVideoCleaner(use_s3=use_s3)

    def _download_video_from_s3(self, s3_key: str) -> Optional[str]:
        """Download video from S3 to temporary file"""
        if not self.use_s3:
            return None
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Download from S3
            success = default_s3_manager.download_file(s3_key, temp_path)
            if success:
                logger.debug(f"Downloaded video from S3 to temp file: {temp_path}")
                return temp_path
            else:
                # Clean up failed download
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None

        except Exception as e:
            logger.error(f"Error downloading video from S3: {e}")
            return None

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {temp_path}: {e}")

    def _upload_frame_to_s3(
        self, local_path: str, game_name: str, video_id: str, frame_filename: str
    ) -> bool:
        """Upload a frame to S3"""
        if not self.use_s3:
            return False

        try:
            # Create S3 key
            game_dir = game_name.replace(" ", "_").lower()
            s3_key = f"{self.frames_s3_prefix}/{game_dir}/{video_id}/{frame_filename}"

            # Upload frame
            success = default_s3_manager.upload_file(local_path, s3_key)
            if success:
                logger.debug(f"Uploaded frame to S3: {s3_key}")
                return True
            else:
                logger.warning(f"Failed to upload frame to S3: {s3_key}")
                return False

        except Exception as e:
            logger.error(f"Error uploading frame to S3: {e}")
            return False

    def _upload_metadata_to_s3(
        self,
        metadata_dict: Dict[str, Any],
        game_name: str,
        video_id: str,
        filename: str,
    ) -> bool:
        """Upload metadata to S3"""
        if not self.use_s3:
            return False

        try:
            # Create S3 key
            game_dir = game_name.replace(" ", "_").lower()
            s3_key = f"{self.frames_s3_prefix}/{game_dir}/{video_id}/{filename}"

            # Upload metadata
            success = default_s3_manager.upload_json(metadata_dict, s3_key)
            if success:
                logger.debug(f"Uploaded metadata to S3: {s3_key}")
                return True
            else:
                logger.warning(f"Failed to upload metadata to S3: {s3_key}")
                return False

        except Exception as e:
            logger.error(f"Error uploading metadata to S3: {e}")
            return False

    def calculate_target_size(
        self, crop_width: int, crop_height: int
    ) -> Tuple[int, int]:
        """Calculate target size maintaining aspect ratio"""
        aspect_ratio = crop_width / crop_height
        target_width = int(self.target_height * aspect_ratio)

        # Ensure even dimensions for video compatibility
        target_width = target_width + (target_width % 2)
        target_height = self.target_height + (self.target_height % 2)

        return target_width, target_height

    def extract_frames_from_video(
        self,
        video_path: str,
        crop_region: CropRegion,
        output_dir: str,
        game_name: str,
        keep_local_frames: bool = True,
    ) -> List[str]:
        """Extract frames from a video with cropping and resizing, supporting S3 storage"""
        temp_file = None
        actual_video_path = video_path

        # Check if this is an S3 path and download if needed
        if self.use_s3 and (
            video_path.startswith("s3://") or not os.path.exists(video_path)
        ):
            # Assume it's an S3 key if not a local file
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

            # Calculate frame sampling
            frame_interval = max(1, int(original_fps / self.target_fps))

            # Skip first and last 10% of video
            start_frame = int(total_frames * 0.1)
            end_frame = int(total_frames * 0.9)

            # Calculate target size
            target_width, target_height = self.calculate_target_size(
                crop_region.width, crop_region.height
            )

            # Create output directory
            video_id = Path(video_path).stem
            video_output_dir = (
                Path(output_dir) / game_name.replace(" ", "_").lower() / video_id
            )
            video_output_dir.mkdir(parents=True, exist_ok=True)

            extracted_frames = []
            frame_count = 0

            # Extract frames
            for frame_idx in range(start_frame, end_frame, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Apply crop
                x, y, w, h = (
                    crop_region.x,
                    crop_region.y,
                    crop_region.width,
                    crop_region.height,
                )
                cropped_frame = frame[y : y + h, x : x + w]

                if cropped_frame.size == 0:
                    continue

                # Resize to target resolution
                resized_frame = cv2.resize(
                    cropped_frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )

                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Save as PNG
                timestamp = frame_idx / original_fps
                frame_filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.png"
                frame_path = video_output_dir / frame_filename

                # Use PIL for better PNG compression
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(frame_path, "PNG", optimize=True)

                # Upload to S3 if enabled
                if self.use_s3:
                    upload_success = self._upload_frame_to_s3(
                        str(frame_path), game_name, video_id, frame_filename
                    )
                    if upload_success and not keep_local_frames:
                        # Remove local file if uploaded successfully and not keeping local copies
                        try:
                            frame_path.unlink()
                        except Exception as e:
                            logger.warning(
                                f"Error removing local frame {frame_path}: {e}"
                            )

                # Create metadata
                metadata = FrameMetadata(
                    video_id=video_id,
                    frame_number=frame_count,
                    timestamp=timestamp,
                    game=game_name,
                    original_resolution=(original_width, original_height),
                    cropped_resolution=(crop_region.width, crop_region.height),
                    final_resolution=(target_width, target_height),
                )

                # Save metadata
                metadata_filename = f"frame_{frame_count:06d}_metadata.json"
                metadata_path = video_output_dir / metadata_filename
                metadata_dict = self._frame_metadata_to_dict(metadata)
                self._save_frame_metadata(metadata, metadata_path)

                # Upload metadata to S3 if enabled
                if self.use_s3:
                    self._upload_metadata_to_s3(
                        metadata_dict, game_name, video_id, metadata_filename
                    )
                    if not keep_local_frames:
                        # Remove local metadata file if uploaded successfully
                        try:
                            metadata_path.unlink()
                        except Exception as e:
                            logger.warning(
                                f"Error removing local metadata {metadata_path}: {e}"
                            )

                extracted_frames.append(str(frame_path))
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Extracted {frame_count} frames from {video_id}")

            cap.release()

            # Save video summary
            self._save_video_summary(
                video_path,
                crop_region,
                game_name,
                frame_count,
                video_output_dir,
                keep_local_frames,
            )

            logger.info(f"Extracted {frame_count} frames from {video_id}")
            return extracted_frames

        finally:
            # Clean up temporary file if we downloaded from S3
            if temp_file:
                self._cleanup_temp_file(temp_file)

    def _frame_metadata_to_dict(self, metadata: FrameMetadata) -> Dict[str, Any]:
        """Convert FrameMetadata to dictionary"""
        return {
            "video_id": metadata.video_id,
            "frame_number": metadata.frame_number,
            "timestamp": metadata.timestamp,
            "game": metadata.game,
            "original_resolution": metadata.original_resolution,
            "cropped_resolution": metadata.cropped_resolution,
            "final_resolution": metadata.final_resolution,
        }

    def _save_frame_metadata(self, metadata: FrameMetadata, metadata_path: Path):
        """Save frame metadata to JSON"""
        metadata_dict = self._frame_metadata_to_dict(metadata)
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _save_video_summary(
        self,
        video_path: str,
        crop_region: CropRegion,
        game_name: str,
        frame_count: int,
        output_dir: Path,
        keep_local: bool = True,
    ):
        """Save summary information for the processed video"""
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
            },
            "results": {
                "frames_extracted": frame_count,
                "output_directory": str(output_dir),
            },
            "storage": {"use_s3": self.use_s3, "keep_local_frames": keep_local},
        }

        summary_path = output_dir / "video_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Upload summary to S3 if enabled
        if self.use_s3:
            video_id = Path(video_path).stem
            self._upload_metadata_to_s3(
                summary, game_name, video_id, "video_summary.json"
            )

    def process_video(
        self,
        video_path: str,
        game_name: str,
        output_dir: str,
        keep_local_frames: bool = True,
    ) -> bool:
        """Process a single video: clean, crop, and extract frames"""

        logger.info(f"Processing video: {video_path}")

        # Check if video is clean and get crop region
        is_clean, crop_region = self.cleaner.is_video_clean(video_path)

        if not is_clean or not crop_region:
            logger.warning(f"Video failed cleaning validation: {video_path}")
            return False

        # Extract frames
        try:
            extracted_frames = self.extract_frames_from_video(
                video_path, crop_region, output_dir, game_name, keep_local_frames
            )

            if extracted_frames:
                logger.info(
                    f"Successfully processed video: {video_path} ({len(extracted_frames)} frames)"
                )
                return True
            else:
                logger.warning(f"No frames extracted from video: {video_path}")
                return False

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage configuration"""
        return {
            "use_s3": self.use_s3,
            "s3_bucket": default_s3_manager.bucket_name,
            "raw_s3_prefix": self.raw_s3_prefix,
            "frames_s3_prefix": self.frames_s3_prefix,
            "target_fps": self.target_fps,
            "target_height": self.target_height,
        }

    def process_video_directory(self, video_dir: str, output_dir: str):
        """Process all videos in a directory"""
        video_path = Path(video_dir)

        if not video_path.exists():
            logger.error(f"Video directory does not exist: {video_dir}")
            return

        # Find all video files
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(video_path.glob(f"**/*{ext}"))

        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return

        logger.info(f"Found {len(video_files)} video files to process")

        processed_count = 0
        failed_count = 0

        for video_file in video_files:
            # Determine game name from directory structure
            game_name = self._determine_game_name(video_file)

            if self.process_video(str(video_file), game_name, output_dir):
                processed_count += 1
            else:
                failed_count += 1
                # Optionally move failed videos to a separate directory
                self._handle_failed_video(video_file)

        logger.info(
            f"Processing complete: {processed_count} successful, {failed_count} failed"
        )

    def _determine_game_name(self, video_path: Path) -> str:
        """Determine Pokemon game name from video path"""
        path_str = str(video_path).lower()

        game_mappings = {
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

        for key, game_name in game_mappings.items():
            if key in path_str:
                return game_name

        # Default fallback
        return "Pokemon Unknown"

    def _handle_failed_video(self, video_path: Path):
        """Handle videos that failed processing"""
        failed_dir = video_path.parent / "failed_videos"
        failed_dir.mkdir(exist_ok=True)

        try:
            # Move to failed directory
            new_path = failed_dir / video_path.name
            video_path.rename(new_path)
            logger.info(f"Moved failed video to: {new_path}")
        except Exception as e:
            logger.error(f"Could not move failed video {video_path}: {e}")

    def create_dataset_index(self, output_dir: str):
        """Create an index of all extracted frames for easy dataset loading"""
        output_path = Path(output_dir)

        dataset_index = {
            "games": {},
            "total_frames": 0,
            "extraction_settings": {
                "target_fps": self.target_fps,
                "target_height": self.target_height,
            },
        }

        # Scan all game directories
        for game_dir in output_path.iterdir():
            if not game_dir.is_dir():
                continue

            game_name = game_dir.name
            dataset_index["games"][game_name] = {"videos": {}, "frame_count": 0}

            # Scan all video directories
            for video_dir in game_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                video_id = video_dir.name

                # Count frames
                frame_files = list(video_dir.glob("frame_*.png"))
                frame_count = len(frame_files)

                if frame_count > 0:
                    dataset_index["games"][game_name]["videos"][video_id] = {
                        "frame_count": frame_count,
                        "directory": str(video_dir),
                    }
                    dataset_index["games"][game_name]["frame_count"] += frame_count
                    dataset_index["total_frames"] += frame_count

        # Save index
        index_path = output_path / "dataset_index.json"
        with open(index_path, "w") as f:
            json.dump(dataset_index, f, indent=2)

        logger.info(f"Created dataset index: {index_path}")
        logger.info(f"Total frames in dataset: {dataset_index['total_frames']}")

        return dataset_index


def main():
    """Main function for testing"""
    extractor = PokemonFrameExtractor(target_fps=5, target_height=360)

    # Process videos from raw_videos directory
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
