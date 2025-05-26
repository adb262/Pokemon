#!/usr/bin/env python3
"""
Pokemon Video Scraper
Downloads Pokemon gameplay videos from YouTube for dataset creation.
Focuses on classic games: Emerald, Fire Red, Ruby, Sapphire, Heart Gold, Soul Silver.
Supports both local storage and S3 upload.
"""

from idm.s3_utils import S3Manager, get_s3_manager_from_env
import logging
import yt_dlp
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import time
import sys

# Add the parent directory to the path so we can import from idm
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PokemonVideoScraper:
    """Scrapes Pokemon gameplay videos from YouTube with S3 support"""

    # Target Pokemon games
    POKEMON_GAMES = [
        "Pokemon Emerald",
        "Pokemon Fire Red",
        "Pokemon Ruby",
        "Pokemon Sapphire",
        "Pokemon Heart Gold",
        "Pokemon Soul Silver"
    ]

    def __init__(self, output_dir: str = "raw_videos", max_videos_per_game: int = 50,
                 use_s3: bool = False, s3_manager: Optional[S3Manager] = None,
                 s3_prefix: str = "raw_videos", keep_local_copy: bool = False):
        """
        Initialize the Pokemon video scraper

        Args:
            output_dir: Local output directory
            max_videos_per_game: Maximum videos to download per game
            use_s3: Whether to upload videos to S3
            s3_manager: S3Manager instance (if None and use_s3=True, will create from env)
            s3_prefix: S3 prefix for uploaded videos
            keep_local_copy: Whether to keep local copies when using S3
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_videos_per_game = max_videos_per_game
        self.use_s3 = use_s3
        self.s3_prefix = s3_prefix
        self.keep_local_copy = keep_local_copy

        # Initialize S3 manager if needed
        if use_s3:
            if s3_manager is None:
                try:
                    self.s3_manager = get_s3_manager_from_env()
                    logger.info(f"S3 manager initialized for bucket: {self.s3_manager.bucket_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize S3 manager: {e}")
                    logger.error("Falling back to local storage only")
                    self.use_s3 = False
                    self.s3_manager = None
            else:
                self.s3_manager = s3_manager
        else:
            self.s3_manager = None

        self.downloaded_videos = self._load_downloaded_list()

    def _load_downloaded_list(self) -> set:
        """Load list of already downloaded video IDs from local or S3"""
        downloaded_file_local = self.output_dir / "downloaded_videos.json"

        # Try to load from S3 first if enabled
        if self.use_s3 and self.s3_manager:
            s3_key = f"{self.s3_prefix}/downloaded_videos.json"
            downloaded_data = self.s3_manager.download_json(s3_key)
            if downloaded_data:
                logger.info(f"Loaded downloaded videos list from S3: {len(downloaded_data)} videos")
                return set(downloaded_data)

        # Fall back to local file
        if downloaded_file_local.exists():
            with open(downloaded_file_local, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded downloaded videos list from local file: {len(data)} videos")
                return set(data)

        logger.info("No existing downloaded videos list found")
        return set()

    def _save_downloaded_list(self):
        """Save list of downloaded video IDs to local and/or S3"""
        downloaded_data = list(self.downloaded_videos)

        # Save to S3 if enabled
        if self.use_s3 and self.s3_manager:
            s3_key = f"{self.s3_prefix}/downloaded_videos.json"
            # Convert list to dict format for JSON upload
            data_dict = {"downloaded_videos": downloaded_data}
            success = self.s3_manager.upload_json(data_dict, s3_key)
            if success:
                logger.debug(f"Saved downloaded videos list to S3: {s3_key}")
            else:
                logger.warning(f"Failed to save downloaded videos list to S3: {s3_key}")

        # Save locally (always, as backup)
        downloaded_file_local = self.output_dir / "downloaded_videos.json"
        with open(downloaded_file_local, 'w') as f:
            json.dump(downloaded_data, f)
        logger.debug(f"Saved downloaded videos list locally: {downloaded_file_local}")

    def _upload_video_to_s3(self, local_path: Path, game_name: str) -> bool:
        """Upload a video file to S3"""
        if not self.use_s3 or not self.s3_manager:
            return False

        try:
            # Create S3 key
            game_dir = game_name.replace(" ", "_").lower()
            s3_key = f"{self.s3_prefix}/{game_dir}/{local_path.name}"

            # Upload video file
            success = self.s3_manager.upload_file(str(local_path), s3_key)
            if success:
                logger.info(f"Uploaded video to S3: {s3_key}")

                # Also upload info.json if it exists
                info_file = local_path.with_suffix('.info.json')
                if info_file.exists():
                    info_s3_key = f"{self.s3_prefix}/{game_dir}/{info_file.name}"
                    self.s3_manager.upload_file(str(info_file), info_s3_key)
                    logger.debug(f"Uploaded video info to S3: {info_s3_key}")

                return True
            else:
                logger.error(f"Failed to upload video to S3: {s3_key}")
                return False

        except Exception as e:
            logger.error(f"Error uploading video to S3: {e}")
            return False

    def _cleanup_local_file(self, local_path: Path):
        """Remove local file if not keeping local copies"""
        if not self.keep_local_copy:
            try:
                if local_path.exists():
                    local_path.unlink()
                    logger.debug(f"Removed local file: {local_path}")

                # Also remove info file
                info_file = local_path.with_suffix('.info.json')
                if info_file.exists():
                    info_file.unlink()
                    logger.debug(f"Removed local info file: {info_file}")

            except Exception as e:
                logger.warning(f"Error removing local file {local_path}: {e}")

    def _get_search_queries(self) -> List[str]:
        """Generate search queries for Pokemon games"""
        queries = []
        for game in self.POKEMON_GAMES:
            queries.extend([
                f"{game} full gameplay",
                f"{game} full walkthrough",
            ])
        logger.info(f"Generated {len(queries)} search queries")
        return queries

    def _get_ydl_opts(self, game_dir: Path) -> Dict:
        """Get yt-dlp options for downloading"""
        return {
            'format': 'best[height<=720][ext=mp4]',  # Max 720p MP4
            'outtmpl': str(game_dir / '%(title)s.%(id)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': False,
            'extractaudio': False,
            'audioformat': 'none',
            'quiet': False,
            'verbose': False,
        }

    def _filter_video_info(self, info: Dict) -> bool:
        """Filter videos based on metadata"""
        # Duration filters (10 minutes to 8 hours)
        duration = info.get('duration', 0)
        if duration < 600 or duration > 28800:
            logger.info(f"Skipping video {info.get('id')} - duration {duration}s outside range")
            return False

        # Title filters - avoid obvious non-gameplay content
        title = info.get('title', '').lower()
        bad_keywords = [
            'reaction', 'review', 'tier list', 'ranking', 'top 10', 'best',
            'worst', 'analysis', 'explained', 'theory', 'music', 'soundtrack',
            'opening', 'trailer', 'commercial', 'ad', 'speedrun', 'tas',
            'glitch', 'hack', 'rom hack', 'mod', 'randomizer'
        ]

        if any(keyword in title for keyword in bad_keywords):
            logger.info(f"Skipping video {info.get('id')} - contains bad keyword in title")
            return False

        # Must contain gameplay indicators
        good_keywords = ['gameplay', 'walkthrough']
        if not any(keyword in title for keyword in good_keywords):
            logger.info(f"Skipping video {info.get('id')} - no gameplay keywords in title")
            return False

        return True

    def search_and_download_videos(self, query: str, game_name: str, max_results: int = 20) -> List[str]:
        """Search for and download videos for a specific query"""
        game_dir = self.output_dir / game_name.replace(" ", "_").lower()
        game_dir.mkdir(exist_ok=True)

        downloaded_files = []

        # Search options
        search_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
        }

        search_query = f"ytsearch{max_results}:{query}"

        try:
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                logger.info(f"Searching for: {query}")
                search_results = ydl.extract_info(search_query, download=False)

                if not search_results or 'entries' not in search_results:
                    logger.warning(f"No results found for query: {query}")
                    return downloaded_files

                # Download each video
                download_opts = self._get_ydl_opts(game_dir)

                for entry in search_results['entries']:
                    if not entry:
                        continue

                    video_id = entry.get('id')
                    if video_id in self.downloaded_videos:
                        logger.info(f"Video {video_id} already seen, skipping")
                        continue

                    # Get full video info for filtering
                    try:
                        with yt_dlp.YoutubeDL({'quiet': True}) as info_ydl:
                            full_info = info_ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)

                        # Mark video as seen regardless of whether we download it
                        self.downloaded_videos.add(video_id)

                        if full_info and not self._filter_video_info(full_info):
                            title = full_info.get('title', 'Unknown')
                            logger.info(f"Video {title} ({video_id}) filtered out")
                            continue

                        # Download the video
                        with yt_dlp.YoutubeDL(download_opts) as download_ydl:
                            title = full_info.get('title', 'Unknown') if full_info else 'Unknown'
                            logger.info(f"Downloading video: {title} ({video_id})")
                            download_ydl.download([f"https://youtube.com/watch?v={video_id}"])

                        # Find the downloaded file
                        video_files = list(game_dir.glob(f"*.{video_id}.*"))
                        video_file = None
                        for f in video_files:
                            if f.suffix in ['.mp4', '.mkv', '.webm']:
                                video_file = f
                                break

                        if video_file and video_file.exists():
                            # Upload to S3 if enabled
                            if self.use_s3:
                                upload_success = self._upload_video_to_s3(video_file, game_name)
                                if upload_success:
                                    logger.info(f"Successfully uploaded {video_file.name} to S3")
                                    # Clean up local file if not keeping copies
                                    self._cleanup_local_file(video_file)
                                else:
                                    logger.warning(f"Failed to upload {video_file.name} to S3, keeping local copy")

                            downloaded_files.append(video_id)
                        else:
                            logger.warning(f"Downloaded video file not found for {video_id}")

                        # Save progress
                        self._save_downloaded_list()

                        # Rate limiting
                        time.sleep(2)

                    except Exception as e:
                        logger.error(f"Error processing video {video_id}: {e}")
                        # Still mark as seen to avoid retrying
                        self.downloaded_videos.add(video_id)
                        continue

        except Exception as e:
            logger.error(f"Error searching for query '{query}': {e}")

        return downloaded_files

    def scrape_all_games(self):
        """Scrape videos for all Pokemon games"""
        logger.info("Starting Pokemon video scraping...")
        if self.use_s3:
            logger.info(f"S3 upload enabled - bucket: {self.s3_manager.bucket_name if self.s3_manager else 'Unknown'}")
            logger.info(f"Keep local copies: {self.keep_local_copy}")

        total_downloaded = 0

        for game in self.POKEMON_GAMES:
            logger.info(f"Scraping videos for {game}")
            game_downloaded = 0

            # Generate queries for this game
            for query in self._get_search_queries():
                if game_downloaded >= self.max_videos_per_game:
                    break

                remaining = self.max_videos_per_game - game_downloaded
                max_results = min(20, remaining)

                downloaded = self.search_and_download_videos(query, game, max_results)
                game_downloaded += len(downloaded)
                total_downloaded += len(downloaded)

                logger.info(f"Downloaded {len(downloaded)} videos for query: {query}")

                # Rate limiting between queries
                time.sleep(5)

            logger.info(f"Total downloaded for {game}: {game_downloaded}")

        logger.info(f"Scraping complete! Total videos downloaded: {total_downloaded}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage configuration"""
        return {
            'use_s3': self.use_s3,
            's3_bucket': self.s3_manager.bucket_name if self.s3_manager else None,
            's3_prefix': self.s3_prefix,
            'keep_local_copy': self.keep_local_copy,
            'local_output_dir': str(self.output_dir),
            'total_downloaded': len(self.downloaded_videos)
        }


def main():
    """Main function to run the scraper"""
    import argparse

    parser = argparse.ArgumentParser(description='Pokemon Video Scraper with S3 support')
    parser.add_argument('--output-dir', default='raw_videos', help='Local output directory')
    parser.add_argument('--max-videos', type=int, default=30, help='Max videos per game')
    parser.add_argument('--use-s3', action='store_true', help='Upload videos to S3')
    parser.add_argument('--s3-prefix', default='raw_videos', help='S3 prefix for videos')
    parser.add_argument('--keep-local', action='store_true', help='Keep local copies when using S3')

    args = parser.parse_args()

    scraper = PokemonVideoScraper(
        output_dir=args.output_dir,
        max_videos_per_game=args.max_videos,
        use_s3=args.use_s3,
        s3_prefix=args.s3_prefix,
        keep_local_copy=args.keep_local
    )

    # Print configuration
    storage_info = scraper.get_storage_info()
    logger.info("Scraper Configuration:")
    for key, value in storage_info.items():
        logger.info(f"  {key}: {value}")

    scraper.scrape_all_games()


if __name__ == "__main__":
    main()
