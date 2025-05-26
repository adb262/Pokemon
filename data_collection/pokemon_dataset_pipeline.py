"""
Pokemon Dataset Pipeline
Main script that orchestrates the entire Pokemon video dataset creation process:
1. Scrape Pokemon gameplay videos from YouTube
2. Clean and validate videos (remove commentary/overlays)
3. Extract frames at 5fps and 360p resolution
4. Create organized dataset structure
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json
import time

from data_collection.scrape_videos import PokemonVideoScraper
from data_collection.video_cleaner import PokemonVideoCleaner
from data_collection.frame_extractor import PokemonFrameExtractor
from data_collection.data_config import PokemonDatasetPipelineConfig, parse_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pokemon_dataset_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PokemonDatasetPipeline:
    """Main pipeline for creating Pokemon video dataset"""

    def __init__(self, config: PokemonDatasetPipelineConfig):
        self._raw_videos_dir = config.raw_videos_dir
        self._clean_videos_dir = config.clean_videos_dir
        self._frames_dir = config.frames_dir
        self._logs_dir = config.logs_dir
        self._max_videos_per_game = config.max_videos_per_game
        self._min_gameplay_ratio = config.min_gameplay_ratio
        self._target_fps = config.target_fps
        self._target_height = config.target_height
        self.scraper = PokemonVideoScraper(
            output_dir=self._raw_videos_dir,
            max_videos_per_game=self._max_videos_per_game
        )
        self.cleaner = PokemonVideoCleaner(
            min_gameplay_ratio=self._min_gameplay_ratio
        )
        self.extractor = PokemonFrameExtractor(
            target_fps=self._target_fps,
            target_height=self._target_height
        )
        self._scrape = config.scrape
        self._clean = config.clean
        self._extract = config.extract
        self._summary = config.summary

        # Create output directories
        self._create_directories()

    def _get_config(self) -> PokemonDatasetPipelineConfig:
        """Get the configuration for the pipeline"""
        return PokemonDatasetPipelineConfig(
            raw_videos_dir=self._raw_videos_dir,
            clean_videos_dir=self._clean_videos_dir,
            frames_dir=self._frames_dir,
            logs_dir=self._logs_dir,
            max_videos_per_game=self._max_videos_per_game,
            min_gameplay_ratio=self._min_gameplay_ratio,
            target_fps=self._target_fps,
            target_height=self._target_height,
            scrape=self._scrape,
        )

    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            self._raw_videos_dir,
            self._clean_videos_dir,
            self._frames_dir,
            self._logs_dir
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def step1_scrape_videos(self) -> bool:
        """Step 1: Scrape Pokemon videos from YouTube"""
        logger.info("=" * 60)
        logger.info("STEP 1: Scraping Pokemon videos from YouTube")
        logger.info("=" * 60)

        try:
            self.scraper.scrape_all_games()
            logger.info("Video scraping completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in video scraping: {e}")
            return False

    def step2_clean_videos(self) -> bool:
        """Step 2: Clean and validate scraped videos"""
        logger.info("=" * 60)
        logger.info("STEP 2: Cleaning and validating videos")
        logger.info("=" * 60)

        try:
            raw_videos_path = Path(self._raw_videos_dir)
            clean_videos_path = Path(self._clean_videos_dir)

            if not raw_videos_path.exists():
                logger.error(f"Raw videos directory does not exist: {raw_videos_path}")
                return False

            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
            video_files = []

            for ext in video_extensions:
                video_files.extend(raw_videos_path.glob(f"**/*{ext}"))

            if not video_files:
                logger.warning("No video files found to clean")
                return True

            logger.info(f"Found {len(video_files)} videos to analyze")

            clean_count = 0
            rejected_count = 0

            # Process each video
            for video_file in video_files:
                logger.info(f"Analyzing video: {video_file.name}")

                is_clean, crop_region = self.cleaner.is_video_clean(str(video_file))

                if is_clean and crop_region:
                    # Save crop information
                    game_name = self._determine_game_from_path(video_file)
                    crop_dir = clean_videos_path / game_name.replace(" ", "_").lower()
                    crop_dir.mkdir(parents=True, exist_ok=True)

                    self.cleaner.save_crop_info(
                        str(video_file), crop_region, str(crop_dir)
                    )

                    clean_count += 1
                    logger.info(f"✓ Video approved: {video_file.name}")
                else:
                    rejected_count += 1
                    logger.info(f"✗ Video rejected: {video_file.name}")

                    # Move rejected video
                    self._move_rejected_video(video_file)

            logger.info(f"Video cleaning completed: {clean_count} clean, {rejected_count} rejected")
            return True

        except Exception as e:
            logger.error(f"Error in video cleaning: {e}")
            return False

    def step3_extract_frames(self) -> bool:
        """Step 3: Extract frames from clean videos"""
        logger.info("=" * 60)
        logger.info("STEP 3: Extracting frames from clean videos")
        logger.info("=" * 60)

        try:
            # Process all videos in raw_videos directory
            self.extractor.process_video_directory(
                self._raw_videos_dir,
                self._frames_dir
            )

            # Create dataset index
            dataset_index = self.extractor.create_dataset_index(self._frames_dir)

            logger.info("Frame extraction completed successfully")
            logger.info(f"Total frames extracted: {dataset_index['total_frames']}")

            return True

        except Exception as e:
            logger.error(f"Error in frame extraction: {e}")
            return False

    def step4_create_dataset_summary(self) -> bool:
        """Step 4: Create final dataset summary and statistics"""
        logger.info("=" * 60)
        logger.info("STEP 4: Creating dataset summary")
        logger.info("=" * 60)

        try:
            frames_path = Path(self._frames_dir)

            # Load dataset index
            index_file = frames_path / "dataset_index.json"
            if not index_file.exists():
                logger.error("Dataset index not found")
                return False

            with open(index_file, 'r') as f:
                dataset_index = json.load(f)

            # Create comprehensive summary
            summary = {
                **self._get_config().__dict__,
                'dataset_statistics': dataset_index,
                'creation_timestamp': time.time(),
                'games_included': list(dataset_index['games'].keys()),
                'total_videos_processed': sum(
                    len(game_data['videos'])
                    for game_data in dataset_index['games'].values()
                ),
                'total_frames': dataset_index['total_frames'],
                'average_frames_per_video': 0,
                'storage_info': self._calculate_storage_info(frames_path)
            }

            # Calculate average frames per video
            total_videos = summary['total_videos_processed']
            if total_videos > 0:
                summary['average_frames_per_video'] = summary['total_frames'] / total_videos

            # Save summary
            summary_file = frames_path / "dataset_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Print summary
            self._print_final_summary(summary)

            logger.info(f"Dataset summary saved to: {summary_file}")
            return True

        except Exception as e:
            logger.error(f"Error creating dataset summary: {e}")
            return False

    def _determine_game_from_path(self, video_path: Path) -> str:
        """Determine Pokemon game from video path"""
        path_str = str(video_path).lower().replace(" ", "_")

        game_mappings = {
            'emerald': 'Pokemon Emerald',
            'fire_red': 'Pokemon Fire Red',
            'firered': 'Pokemon Fire Red',
            'ruby': 'Pokemon Ruby',
            'sapphire': 'Pokemon Sapphire',
            'heart_gold': 'Pokemon Heart Gold',
            'heartgold': 'Pokemon Heart Gold',
            'soul_silver': 'Pokemon Soul Silver',
            'soulsilver': 'Pokemon Soul Silver'
        }

        for key, game_name in game_mappings.items():
            if key in path_str:
                return game_name

        return "Pokemon Unknown"

    def _move_rejected_video(self, video_path: Path):
        """Move rejected video to rejected directory"""
        rejected_dir = video_path.parent / "rejected_videos"
        rejected_dir.mkdir(exist_ok=True)

        try:
            new_path = rejected_dir / video_path.name
            video_path.rename(new_path)
            logger.debug(f"Moved rejected video to: {new_path}")
        except Exception as e:
            logger.warning(f"Could not move rejected video {video_path}: {e}")

    def _calculate_storage_info(self, frames_path: Path) -> Dict[str, Any]:
        """Calculate storage information for the dataset"""
        total_size = 0
        total_files = 0

        for file_path in frames_path.rglob("*.png"):
            try:
                total_size += file_path.stat().st_size
                total_files += 1
            except OSError:
                continue

        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
            'total_files': total_files,
            'average_file_size_kb': round(total_size / total_files / 1024, 2) if total_files > 0 else 0
        }

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final dataset summary"""
        logger.info("\n" + "=" * 80)
        logger.info("POKEMON DATASET CREATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total Games: {len(summary['games_included'])}")
        logger.info(f"Games: {', '.join(summary['games_included'])}")
        logger.info(f"Total Videos Processed: {summary['total_videos_processed']}")
        logger.info(f"Total Frames Extracted: {summary['total_frames']:,}")
        logger.info(f"Average Frames per Video: {summary['average_frames_per_video']:.1f}")
        logger.info(f"Dataset Size: {summary['storage_info']['total_size_gb']:.2f} GB")
        logger.info(f"Total Files: {summary['storage_info']['total_files']:,}")
        logger.info(f"Average File Size: {summary['storage_info']['average_file_size_kb']:.1f} KB")
        logger.info("=" * 80)

    def run_full_pipeline(self) -> bool:
        """Run the complete dataset creation pipeline"""
        logger.info("Starting Pokemon Dataset Creation Pipeline")
        logger.info(f"Configuration: {self._get_config()}")

        start_time = time.time()

        # Step 1: Scrape videos
        if self._scrape and not self.step1_scrape_videos():
            logger.error("Pipeline failed at step 1 (video scraping)")
            return False

        # Step 2: Clean videos
        if self._clean and not self.step2_clean_videos():
            logger.error("Pipeline failed at step 2 (video cleaning)")
            return False

        # Step 3: Extract frames
        if self._extract and not self.step3_extract_frames():
            logger.error("Pipeline failed at step 3 (frame extraction)")
            return False

        # Step 4: Create summary
        if self._summary and not self.step4_create_dataset_summary():
            logger.error("Pipeline failed at step 4 (dataset summary)")
            return False

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Pipeline completed successfully in {duration:.1f} seconds")
        return True


def main():
    """Main function with command line interface"""
    args = parse_args()
    # Create and run pipeline
    pipeline = PokemonDatasetPipeline(args)
    success = pipeline.run_full_pipeline()
    logger.info(f"Pipeline completed with success: {success}")


if __name__ == "__main__":
    main()
