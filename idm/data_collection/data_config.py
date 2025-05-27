import argparse
from dataclasses import dataclass


@dataclass
class PokemonDatasetPipelineConfig:
    raw_videos_dir: str = 'raw_videos'
    clean_videos_dir: str = 'clean_videos'
    frames_dir: str = 'pokemon_frames'
    logs_dir: str = 'logs'
    max_videos_per_game: int = 30
    scrape: bool = True
    clean: bool = True
    extract: bool = True
    summary: bool = True
    min_gameplay_ratio: float = 0.6
    target_fps: int = 5
    target_height: int = 360
    # S3 configuration
    use_s3: bool = False
    s3_raw_prefix: str = 'raw_videos'
    s3_clean_prefix: str = 'clean_videos'
    s3_frames_prefix: str = 'pokemon_frames'
    keep_local_videos: bool = True
    keep_local_frames: bool = True


def parse_args() -> PokemonDatasetPipelineConfig:
    default_config = PokemonDatasetPipelineConfig()
    parser = argparse.ArgumentParser(description='Pokemon Dataset Creation Pipeline with S3 support')
    parser.add_argument('--max-videos', type=int, default=default_config.max_videos_per_game,
                        help='Maximum videos per game to download')
    parser.add_argument('--target-fps', type=int, default=default_config.target_fps,
                        help='Target FPS for frame extraction')
    parser.add_argument('--target-height', type=int, default=default_config.target_height,
                        help='Target height for frame extraction')
    parser.add_argument('--min-gameplay-ratio', type=float, default=default_config.min_gameplay_ratio,
                        help='Minimum gameplay ratio for video cleaning')
    parser.add_argument('--raw-videos-dir', type=str, default=default_config.raw_videos_dir,
                        help='Directory to store raw videos')
    parser.add_argument('--clean-videos-dir', type=str, default=default_config.clean_videos_dir,
                        help='Directory to store clean videos')
    parser.add_argument('--frames-dir', type=str, default=default_config.frames_dir,
                        help='Directory to store extracted frames')
    parser.add_argument('--logs-dir', type=str, default=default_config.logs_dir,
                        help='Directory to store logs')
    parser.add_argument('--scrape', action='store_true',
                        help='Scrape videos from YouTube', default=False)
    parser.add_argument('--clean', action='store_true',
                        help='Clean videos', default=False)
    parser.add_argument('--extract', action='store_true',
                        help='Extract frames', default=False)
    parser.add_argument('--summary', action='store_true',
                        help='Create dataset summary', default=True)

    # S3 arguments
    parser.add_argument('--use-s3', action='store_true',
                        help='Use S3 for storage', default=False)
    parser.add_argument('--s3-raw-prefix', type=str, default=default_config.s3_raw_prefix,
                        help='S3 prefix for raw videos')
    parser.add_argument('--s3-clean-prefix', type=str, default=default_config.s3_clean_prefix,
                        help='S3 prefix for clean video metadata')
    parser.add_argument('--s3-frames-prefix', type=str, default=default_config.s3_frames_prefix,
                        help='S3 prefix for extracted frames')
    parser.add_argument('--keep-local-videos', action='store_true',
                        help='Keep local copies of videos when using S3', default=True)
    parser.add_argument('--keep-local-frames', action='store_true',
                        help='Keep local copies of frames when using S3', default=True)
    parser.add_argument('--no-keep-local-videos', dest='keep_local_videos', action='store_false',
                        help='Do not keep local copies of videos when using S3')
    parser.add_argument('--no-keep-local-frames', dest='keep_local_frames', action='store_false',
                        help='Do not keep local copies of frames when using S3')

    args = parser.parse_args()

    config = PokemonDatasetPipelineConfig(
        max_videos_per_game=args.max_videos,
        target_fps=args.target_fps,
        target_height=args.target_height,
        min_gameplay_ratio=args.min_gameplay_ratio,
        raw_videos_dir=args.raw_videos_dir,
        clean_videos_dir=args.clean_videos_dir,
        frames_dir=args.frames_dir,
        logs_dir=args.logs_dir,
        scrape=args.scrape,
        clean=args.clean,
        extract=args.extract,
        summary=args.summary,
        use_s3=args.use_s3,
        s3_raw_prefix=args.s3_raw_prefix,
        s3_clean_prefix=args.s3_clean_prefix,
        s3_frames_prefix=args.s3_frames_prefix,
        keep_local_videos=getattr(args, 'keep_local_videos', True),
        keep_local_frames=getattr(args, 'keep_local_frames', True),
    )
    return config
