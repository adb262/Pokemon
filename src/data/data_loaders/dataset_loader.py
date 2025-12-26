import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class PokemonDatasetLoader:
    """Loader for Pokemon frame dataset"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.index = self._load_dataset_index()
        self.games = list(self.index["games"].keys()) if self.index else []

    def _load_dataset_index(self) -> Optional[Dict[str, Any]]:
        """Load the dataset index file"""
        index_file = self.dataset_path / "dataset_index.json"

        if not index_file.exists():
            logger.error(f"Dataset index not found: {index_file}")
            return None

        try:
            with open(index_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset index: {e}")
            return None

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        if not self.index:
            return {}

        return {
            "total_frames": self.index["total_frames"],
            "total_games": len(self.index["games"]),
            "games": list(self.index["games"].keys()),
            "total_videos": sum(
                len(game_data["videos"]) for game_data in self.index["games"].values()
            ),
            "extraction_settings": self.index.get("extraction_settings", {}),
            "frames_per_game": {
                game: data["frame_count"] for game, data in self.index["games"].items()
            },
        }

    def get_game_frames(self, game_name: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all frame paths and metadata for a specific game"""
        if not self.index or game_name not in self.index["games"]:
            logger.warning(f"Game not found: {game_name}")
            return []

        frames = []
        game_data = self.index["games"][game_name]

        for video_id, video_data in game_data["videos"].items():
            video_dir = Path(video_data["directory"])

            # Find all frame files
            frame_files = sorted(video_dir.glob("frame_*.png"))

            for frame_file in frame_files:
                # Load metadata if available
                metadata_file = frame_file.parent / f"{frame_file.stem}_metadata.json"
                metadata = {}

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {frame_file}: {e}")

                frames.append((str(frame_file), metadata))

        return frames

    def iterate_frames(
        self, game_filter: Optional[List[str]] = None, shuffle: bool = False
    ) -> Iterator[Tuple[str, Dict[str, Any], np.ndarray]]:
        """Iterate through all frames in the dataset"""
        if not self.index:
            return

        games_to_use = game_filter if game_filter else self.games
        all_frames = []

        for game in games_to_use:
            if game in self.index["games"]:
                all_frames.extend(self.get_game_frames(game))

        if shuffle:
            random.shuffle(all_frames)

        for frame_path, metadata in all_frames:
            try:
                # Load image
                image = Image.open(frame_path)
                image_array = np.array(image)

                yield frame_path, metadata, image_array

            except Exception as e:
                logger.warning(f"Could not load frame {frame_path}: {e}")
                continue

    def get_frames_by_video(self, video_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all frames from a specific video"""
        if not self.index:
            return []

        for game_data in self.index["games"].values():
            if video_id in game_data["videos"]:
                video_data = game_data["videos"][video_id]
                video_dir = Path(video_data["directory"])

                frames = []
                frame_files = sorted(video_dir.glob("frame_*.png"))

                for frame_file in frame_files:
                    metadata_file = (
                        frame_file.parent / f"{frame_file.stem}_metadata.json"
                    )
                    metadata = {}

                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                        except Exception:
                            pass

                    frames.append((str(frame_file), metadata))

                return frames

        logger.warning(f"Video not found: {video_id}")
        return []

    def create_frame_sequence(
        self, video_id: str, max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Create a sequence of frames from a video for temporal analysis"""
        frames = self.get_frames_by_video(video_id)

        if not frames:
            return []

        # Sort by frame number if metadata is available
        def get_frame_number(frame_data):
            _, metadata = frame_data
            return metadata.get("frame_number", 0)

        frames.sort(key=get_frame_number)

        if max_frames:
            frames = frames[:max_frames]

        sequence = []
        for frame_path, _ in frames:
            try:
                image = Image.open(frame_path)
                sequence.append(np.array(image))
            except Exception as e:
                logger.warning(f"Could not load frame {frame_path}: {e}")

        return sequence

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the dataset"""
        if not self.index:
            return {}

        stats = {
            "total_frames": self.index["total_frames"],
            "games": {},
            "resolution_info": {},
            "temporal_info": {},
        }

        # Analyze each game
        for game_name, game_data in self.index["games"].items():
            game_stats = {
                "frame_count": game_data["frame_count"],
                "video_count": len(game_data["videos"]),
                "videos": {},
            }

            # Analyze each video
            for video_id, video_data in game_data["videos"].items():
                video_stats = {
                    "frame_count": video_data["frame_count"],
                    "directory": video_data["directory"],
                }

                # Try to get additional info from first frame metadata
                video_dir = Path(video_data["directory"])
                first_frame_metadata = video_dir / "frame_000000_metadata.json"

                if first_frame_metadata.exists():
                    try:
                        with open(first_frame_metadata, "r") as f:
                            metadata = json.load(f)
                            video_stats.update(
                                {
                                    "original_resolution": metadata.get(
                                        "original_resolution"
                                    ),
                                    "final_resolution": metadata.get(
                                        "final_resolution"
                                    ),
                                    "game": metadata.get("game"),
                                }
                            )
                    except Exception:
                        pass

                game_stats["videos"][video_id] = video_stats

            stats["games"][game_name] = game_stats

        return stats

    def export_frame_list(
        self, output_file: str, game_filter: Optional[List[str]] = None
    ):
        """Export a list of all frame paths to a text file"""
        if not self.index:
            logger.error("No dataset index available")
            return

        games_to_use = game_filter if game_filter else self.games

        with open(output_file, "w") as f:
            for game in games_to_use:
                if game in self.index["games"]:
                    frames = self.get_game_frames(game)
                    for frame_path, metadata in frames:
                        f.write(f"{frame_path}\n")

        logger.info(f"Exported frame list to {output_file}")

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the dataset integrity"""
        if not self.index:
            return {"valid": False, "error": "No dataset index found"}

        validation_results = {
            "valid": True,
            "total_frames_expected": self.index["total_frames"],
            "total_frames_found": 0,
            "missing_frames": [],
            "corrupted_frames": [],
            "missing_metadata": [],
            "games_validated": {},
        }

        for game_name, game_data in self.index["games"].items():
            game_validation = {
                "frames_expected": game_data["frame_count"],
                "frames_found": 0,
                "videos_validated": {},
            }

            for video_id, video_data in game_data["videos"].items():
                video_dir = Path(video_data["directory"])
                expected_frames = video_data["frame_count"]

                if not video_dir.exists():
                    validation_results["valid"] = False
                    continue

                # Check frame files
                frame_files = list(video_dir.glob("frame_*.png"))
                found_frames = len(frame_files)

                game_validation["frames_found"] += found_frames
                validation_results["total_frames_found"] += found_frames

                # Validate each frame
                for frame_file in frame_files:
                    try:
                        # Try to open the image
                        with Image.open(frame_file) as img:
                            img.verify()
                    except Exception:
                        validation_results["corrupted_frames"].append(str(frame_file))
                        validation_results["valid"] = False

                    # Check for metadata
                    metadata_file = (
                        frame_file.parent / f"{frame_file.stem}_metadata.json"
                    )
                    if not metadata_file.exists():
                        validation_results["missing_metadata"].append(str(frame_file))

                game_validation["videos_validated"][video_id] = {
                    "expected_frames": expected_frames,
                    "found_frames": found_frames,
                    "valid": found_frames == expected_frames,
                }

                if found_frames != expected_frames:
                    validation_results["valid"] = False

            validation_results["games_validated"][game_name] = game_validation

        # Check total frame count
        if (
            validation_results["total_frames_found"]
            != validation_results["total_frames_expected"]
        ):
            validation_results["valid"] = False

        return validation_results


def main():
    """Example usage of the dataset loader"""
    import argparse

    parser = argparse.ArgumentParser(description="Pokemon Dataset Loader")
    parser.add_argument("dataset_path", help="Path to the Pokemon dataset")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    parser.add_argument(
        "--validate", action="store_true", help="Validate dataset integrity"
    )
    parser.add_argument("--export-list", type=str, help="Export frame list to file")
    parser.add_argument("--game", type=str, help="Filter by specific game")

    args = parser.parse_args()

    # Load dataset
    loader = PokemonDatasetLoader(args.dataset_path)

    if args.info:
        info = loader.get_dataset_info()
        print("Dataset Information:")
        print(f"Total Frames: {info['total_frames']:,}")
        print(f"Total Games: {info['total_games']}")
        print(f"Total Videos: {info['total_videos']}")
        print(f"Games: {', '.join(info['games'])}")
        print("\nFrames per Game:")
        for game, count in info["frames_per_game"].items():
            print(f"  {game}: {count:,}")

    if args.validate:
        print("Validating dataset...")
        results = loader.validate_dataset()
        print(f"Dataset Valid: {results['valid']}")
        print(f"Total Frames Found: {results['total_frames_found']:,}")
        if not results["valid"]:
            print(f"Corrupted Frames: {len(results['corrupted_frames'])}")
            print(f"Missing Metadata: {len(results['missing_metadata'])}")

    if args.export_list:
        game_filter = [args.game] if args.game else None
        loader.export_frame_list(args.export_list, game_filter)
        print(f"Frame list exported to {args.export_list}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
