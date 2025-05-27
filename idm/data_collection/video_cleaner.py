#!/usr/bin/env python3
"""
Pokemon Video Cleaner
Analyzes videos to detect and crop out commentary overlays, UI elements, and non-gameplay content.
Ensures only clean Pokemon gameplay footage is retained.
Supports both local storage and S3.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from s3_utils import default_s3_manager

# Add the parent directory to the path so we can import from idm
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


@dataclass
class CropRegion:
    """Represents a crop region for a video"""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PokemonVideoCleaner:
    """Cleans Pokemon videos by detecting and removing commentary/UI overlays with S3 support"""

    def __init__(self, min_gameplay_ratio: float = 0.6, use_s3: bool = False, s3_prefix: str = "raw_videos",
                 clean_s3_prefix: str = "clean_videos"):
        """
        Initialize the Pokemon video cleaner

        Args:
            min_gameplay_ratio: Minimum ratio of gameplay content required
            use_s3: Whether to use S3 for storage
            s3_manager: S3Manager instance (if None and use_s3=True, will create from env)
            s3_prefix: S3 prefix for raw videos
            clean_s3_prefix: S3 prefix for clean video metadata
        """
        self.min_gameplay_ratio = min_gameplay_ratio
        self.use_s3 = use_s3
        self.s3_prefix = s3_prefix
        self.clean_s3_prefix = clean_s3_prefix
        self.pokemon_colors = self._get_pokemon_game_colors()

    def _download_video_from_s3(self, s3_key: str) -> Optional[str]:
        """Download video from S3 to temporary file"""
        if not self.use_s3:
            return None

        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
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

    def _upload_crop_info_to_s3(self, crop_region: CropRegion, video_path: str, game_name: str) -> bool:
        """Upload crop information to S3"""
        if not self.use_s3:
            return False

        try:
            # Create crop info data
            crop_info = {
                'source_video': video_path,
                'game': game_name,
                'crop_region': {
                    'x': crop_region.x,
                    'y': crop_region.y,
                    'width': crop_region.width,
                    'height': crop_region.height,
                    'confidence': crop_region.confidence
                },
                'analysis_settings': {
                    'min_gameplay_ratio': self.min_gameplay_ratio
                }
            }

            # Create S3 key
            video_name = Path(video_path).stem
            game_dir = game_name.replace(" ", "_").lower()
            s3_key = f"{self.clean_s3_prefix}/{game_dir}/{video_name}_crop_info.json"

            # Upload to S3
            success = default_s3_manager.upload_json(crop_info, s3_key)
            if success:
                logger.info(f"Uploaded crop info to S3: {s3_key}")
                return True
            else:
                logger.error(f"Failed to upload crop info to S3: {s3_key}")
                return False

        except Exception as e:
            logger.error(f"Error uploading crop info to S3: {e}")
            return False

    def _get_pokemon_game_colors(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Get characteristic colors for each Pokemon game for validation"""
        return {
            'emerald': [(34, 139, 34), (0, 100, 0), (144, 238, 144)],  # Green tones
            'fire_red': [(178, 34, 34), (255, 69, 0), (220, 20, 60)],  # Red tones
            'ruby': [(178, 34, 34), (139, 0, 0), (205, 92, 92)],       # Ruby red
            'sapphire': [(0, 0, 139), (30, 144, 255), (70, 130, 180)],  # Blue tones
            'heart_gold': [(255, 215, 0), (255, 140, 0), (255, 165, 0)],  # Gold tones
            'soul_silver': [(192, 192, 192), (169, 169, 169), (211, 211, 211)]  # Silver tones
        }

    def analyze_video_structure(self, video_path: str, sample_frames: int = 30) -> Optional[CropRegion]:
        """
        Analyze video to detect consistent borders/overlays that should be cropped out.
        Returns the optimal crop region for gameplay content.
        Supports both local files and S3 videos.
        """
        temp_file = None
        actual_video_path = video_path

        # Check if this is an S3 path and download if needed
        if self.use_s3 and (video_path.startswith('s3://') or not os.path.exists(video_path)):
            # Assume it's an S3 key if not a local file
            s3_key = video_path if not video_path.startswith('s3://') else video_path[5:].split('/', 1)[1]
            temp_file = self._download_video_from_s3(s3_key)
            if not temp_file:
                logger.error(f"Could not download video from S3: {s3_key}")
                return None
            actual_video_path = temp_file

        try:
            cap = cv2.VideoCapture(actual_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {actual_video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            if duration < 300:  # Less than 5 minutes
                logger.info(f"Video too short ({duration:.1f}s), skipping: {video_path}")
                cap.release()
                return None

            # Sample frames throughout the video (skip first/last 10%)
            start_frame = int(total_frames * 0.1)
            end_frame = int(total_frames * 0.9)
            frame_indices = np.linspace(start_frame, end_frame, sample_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            cap.release()

            if len(frames) < sample_frames // 2:
                logger.warning(f"Could not extract enough frames from: {video_path}")
                return None

            return self._detect_gameplay_region(frames, video_path)

        finally:
            # Clean up temporary file if we downloaded from S3
            if temp_file:
                self._cleanup_temp_file(temp_file)

    def _detect_gameplay_region(self, frames: List[np.ndarray], video_path: str) -> Optional[CropRegion]:
        """Detect the main gameplay region by analyzing frame consistency"""
        if not frames:
            return None

        height, width = frames[0].shape[:2]

        # Convert frames to grayscale for edge detection
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        # Detect edges in each frame
        edge_frames = [cv2.Canny(gray, 50, 150) for gray in gray_frames]

        # Sum all edge maps to find consistent edges (likely UI elements)
        edge_sum = np.zeros_like(edge_frames[0], dtype=np.float32)
        for edge_frame in edge_frames:
            edge_sum += edge_frame.astype(np.float32)

        edge_sum /= len(edge_frames)

        # Find regions with consistent edges (likely static UI)
        consistent_edges = edge_sum > (0.3 * 255)  # Edges present in 30%+ of frames

        # Analyze horizontal and vertical projections to find borders
        h_projection = np.mean(consistent_edges, axis=1)
        v_projection = np.mean(consistent_edges, axis=0)

        # Find the largest rectangular region with minimal consistent edges
        crop_region = self._find_optimal_crop_region(
            consistent_edges, h_projection, v_projection, width, height
        )

        if crop_region:
            # Validate the crop region contains actual gameplay
            if self._validate_gameplay_content(frames, crop_region):
                logger.info(f"Found valid crop region for {video_path}: {crop_region}")
                return crop_region
            else:
                logger.warning(f"Crop region validation failed for {video_path}")
                return None

        # If no clear crop region found, check if the full frame is valid gameplay
        full_region = CropRegion(0, 0, width, height, 0.5)
        if self._validate_gameplay_content(frames, full_region):
            logger.info(f"Using full frame for {video_path}")
            return full_region

        logger.warning(f"No valid gameplay region found for {video_path}")
        return None

    def _find_optimal_crop_region(self, consistent_edges: np.ndarray,
                                  h_projection: np.ndarray, v_projection: np.ndarray,
                                  width: int, height: int) -> Optional[CropRegion]:
        """Find the optimal rectangular crop region"""

        # Find borders based on edge density
        edge_threshold = 0.1

        # Top border
        top = 0
        for i in range(height // 4):  # Check first quarter
            if h_projection[i] < edge_threshold:
                top = i
                break

        # Bottom border
        bottom = height
        for i in range(height - 1, 3 * height // 4, -1):  # Check last quarter
            if h_projection[i] < edge_threshold:
                bottom = i + 1
                break

        # Left border
        left = 0
        for i in range(width // 4):  # Check first quarter
            if v_projection[i] < edge_threshold:
                left = i
                break

        # Right border
        right = width
        for i in range(width - 1, 3 * width // 4, -1):  # Check last quarter
            if v_projection[i] < edge_threshold:
                right = i + 1
                break

        crop_width = right - left
        crop_height = bottom - top

        # Ensure minimum size and reasonable aspect ratio
        min_width = width * 0.4
        min_height = height * 0.4

        if crop_width < min_width or crop_height < min_height:
            return None

        # Check aspect ratio (Pokemon games are typically 4:3 or 16:9)
        aspect_ratio = crop_width / crop_height
        if aspect_ratio < 1.0 or aspect_ratio > 2.0:
            return None

        confidence = 1.0 - (crop_width * crop_height) / (width * height)

        return CropRegion(left, top, crop_width, crop_height, confidence)

    def _validate_gameplay_content(self, frames: List[np.ndarray], crop_region: CropRegion) -> bool:
        """Validate that the crop region contains actual Pokemon gameplay"""

        # Extract cropped regions
        cropped_frames = []
        for frame in frames[:10]:  # Check first 10 frames
            x, y, w, h = crop_region.x, crop_region.y, crop_region.width, crop_region.height
            cropped = frame[y:y+h, x:x+w]
            if cropped.size > 0:
                cropped_frames.append(cropped)

        if not cropped_frames:
            return False

        # Check for movement/changes between frames (indicates gameplay)
        movement_score = self._calculate_movement_score(cropped_frames)

        # Check for Pokemon-like colors
        color_score = self._calculate_pokemon_color_score(cropped_frames)

        # Check for UI elements typical in Pokemon games
        ui_score = self._detect_pokemon_ui_elements(cropped_frames)

        # Combined score
        total_score = (movement_score * 0.4 + color_score * 0.3 + ui_score * 0.3)

        logger.debug(
            f"Validation scores - Movement: {movement_score:.2f}, Color: {color_score:.2f}, UI: {ui_score:.2f}, Total: {total_score:.2f}")

        return total_score > 0.5

    def _calculate_movement_score(self, frames: List[np.ndarray]) -> float:
        """Calculate movement score between frames"""
        if len(frames) < 2:
            return 0.0

        movement_scores = []
        for i in range(1, len(frames)):
            # Calculate frame difference
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray1, gray2)
            movement = np.mean(diff) / 255.0
            movement_scores.append(movement)

        avg_movement = np.mean(movement_scores)

        # Good gameplay should have moderate movement (not static, not too chaotic)
        if 0.02 < avg_movement < 0.3:
            return 1.0
        elif avg_movement < 0.01:
            return 0.0  # Too static
        else:
            return max(0.0, 1.0 - (avg_movement - 0.3) / 0.7)

    def _calculate_pokemon_color_score(self, frames: List[np.ndarray]) -> float:
        """Calculate how much the frames look like Pokemon gameplay based on colors"""
        if not frames:
            return 0.0

        # Sample colors from frames
        all_colors = []
        for frame in frames[:5]:  # Check first 5 frames
            # Sample pixels from the frame
            h, w = frame.shape[:2]
            sample_points = [(h//4, w//4), (h//2, w//2), (3*h//4, 3*w//4)]

            for y, x in sample_points:
                if y < h and x < w:
                    color = frame[y, x]
                    all_colors.append(tuple(color))

        if not all_colors:
            return 0.0

        # Check for typical Pokemon game colors (greens, blues, earth tones)
        pokemon_like_colors = 0
        total_colors = len(all_colors)

        for color in all_colors:
            b, g, r = color

            # Check for natural/game-like colors
            if (g > r and g > b) or (b > r and b > g) or (50 < r < 200 and 50 < g < 200 and 50 < b < 200):
                pokemon_like_colors += 1

        return pokemon_like_colors / total_colors if total_colors > 0 else 0.0

    def _detect_pokemon_ui_elements(self, frames: List[np.ndarray]) -> float:
        """Detect UI elements typical in Pokemon games"""
        if not frames:
            return 0.0

        ui_score = 0.0
        frame = frames[0]
        h, w = frame.shape[:2]

        # Look for rectangular regions (typical of Pokemon UI)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangular_regions = 0
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular and reasonable size
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if 0.01 * w * h < area < 0.3 * w * h:  # Reasonable size for UI element
                    rectangular_regions += 1

        # Score based on presence of rectangular UI elements
        if rectangular_regions > 0:
            ui_score = min(1.0, rectangular_regions / 5.0)

        return ui_score

    def is_video_clean(self, video_path: str) -> Tuple[bool, Optional[CropRegion]]:
        """
        Determine if a video contains clean Pokemon gameplay.
        Returns (is_clean, crop_region)
        """
        crop_region = self.analyze_video_structure(video_path)

        if crop_region is None:
            return False, None

        # Additional checks for video quality
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, None

        # Check video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        # Minimum quality requirements
        if width < 480 or height < 360:
            logger.info(f"Video resolution too low: {width}x{height}")
            return False, None

        if fps < 15:
            logger.info(f"Video FPS too low: {fps}")
            return False, None

        # Check if crop region is reasonable size
        crop_area_ratio = (crop_region.width * crop_region.height) / (width * height)
        if crop_area_ratio < self.min_gameplay_ratio:
            logger.info(f"Crop region too small: {crop_area_ratio:.2f}")
            return False, None

        return True, crop_region

    def save_crop_info(self, video_path: str, crop_region: CropRegion, output_dir: str, game_name: str = ""):
        """Save crop information to local file and/or S3"""
        # Determine game name if not provided
        if not game_name:
            game_name = self._determine_game_from_path(Path(video_path))

        # Save to S3 if enabled
        if self.use_s3:
            self._upload_crop_info_to_s3(crop_region, video_path, game_name)

        # Save locally (always, as backup)
        crop_info = {
            'source_video': video_path,
            'game': game_name,
            'crop_region': {
                'x': crop_region.x,
                'y': crop_region.y,
                'width': crop_region.width,
                'height': crop_region.height,
                'confidence': crop_region.confidence
            },
            'analysis_settings': {
                'min_gameplay_ratio': self.min_gameplay_ratio
            }
        }

        # Create output file path
        video_name = Path(video_path).stem
        crop_file = Path(output_dir) / f"{video_name}_crop_info.json"

        # Ensure output directory exists
        crop_file.parent.mkdir(parents=True, exist_ok=True)

        with open(crop_file, 'w') as f:
            json.dump(crop_info, f, indent=2)

        logger.info(f"Saved crop info to: {crop_file}")

    def _determine_game_from_path(self, video_path: Path) -> str:
        """Determine game name from video path"""
        path_str = str(video_path).lower()

        for game in ["emerald", "fire_red", "ruby", "sapphire", "heart_gold", "soul_silver"]:
            if game.replace("_", " ") in path_str or game in path_str:
                return game.replace("_", " ").title()

        # Check parent directory
        parent_name = video_path.parent.name.lower()
        for game in ["emerald", "fire_red", "ruby", "sapphire", "heart_gold", "soul_silver"]:
            if game.replace("_", " ") in parent_name or game in parent_name:
                return game.replace("_", " ").title()

        return "Unknown"

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage configuration"""
        return {
            'use_s3': self.use_s3,
            's3_bucket': default_s3_manager.bucket_name,
            's3_prefix': self.s3_prefix,
            'clean_s3_prefix': self.clean_s3_prefix,
            'min_gameplay_ratio': self.min_gameplay_ratio
        }


def main():
    """Test the video cleaner"""
    cleaner = PokemonVideoCleaner()

    # Test with a sample video
    test_video = "raw_videos/pokemon_emerald/sample_video.mp4"
    if os.path.exists(test_video):
        is_clean, crop_region = cleaner.is_video_clean(test_video)
        print(f"Video clean: {is_clean}")
        if crop_region:
            print(f"Crop region: {crop_region}")


if __name__ == "__main__":
    main()
