from dataclasses import dataclass


@dataclass
class PokemonDatasetPipelineConfig:
    raw_videos_dir: str = "raw_videos"
    clean_videos_dir: str = "clean_videos"
    frames_dir: str = "pokemon_frames"
    logs_dir: str = "logs"
    max_videos_per_game: int = 30
    scrape: bool = False
    clean: bool = False
    extract: bool = False
    summary: bool = False
    min_gameplay_ratio: float = 0.6
    target_fps: int = 5
    target_height: int = 360
    # Frame extraction optimization
    jump_seconds: float = 5.0  # Seconds to skip when encountering invalid frames
    num_video_workers: int = 4  # Number of parallel video processing workers
    num_upload_threads: int = 8  # Number of threads for S3 uploads
    # S3 configuration
    use_s3: bool = False
    s3_raw_prefix: str = "raw_videos"
    s3_clean_prefix: str = "clean_videos"
    s3_frames_prefix: str = "pokemon_frames"
    keep_local_videos: bool = True
    keep_local_frames: bool = True
