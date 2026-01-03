#!/usr/bin/env python3
"""
Sync Dataset to S3

Syncs frames from OpenWorldVideoLog dumps to S3.
Supports multithreaded uploads for large datasets.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from data.datasets.data_types.open_world_types import OpenWorldVideoLog
from data.s3.s3_utils import S3Manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of an upload operation."""

    local_path: str
    s3_key: str
    success: bool
    error: Optional[str] = None


@dataclass
class SyncStats:
    """Statistics for the sync operation."""

    total_files: int = 0
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (
            f"Total: {self.total_files}, "
            f"Uploaded: {self.uploaded}, "
            f"Skipped: {self.skipped}, "
            f"Failed: {self.failed}"
        )


class DatasetS3Syncer:
    """Syncs OpenWorldVideoLog datasets to S3 with multithreaded uploads."""

    def __init__(
        self,
        bucket_name: str,
        s3_prefix: Optional[str] = None,
        max_workers: int = 32,
        skip_existing: bool = True,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize the syncer.

        Args:
            bucket_name: S3 bucket to upload to
            s3_prefix: Prefix path in S3 bucket for uploads
            max_workers: Number of concurrent upload threads
            skip_existing: Skip files that already exist in S3
            region_name: AWS region
            aws_access_key_id: AWS access key (uses env vars if not provided)
            aws_secret_access_key: AWS secret key (uses env vars if not provided)
        """
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.rstrip("/") if s3_prefix else None
        self.max_workers = max_workers
        self.skip_existing = skip_existing

        # Initialize S3 manager
        self.s3_manager = S3Manager(
            bucket_name=bucket_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        logger.info(
            f"Initialized DatasetS3Syncer: bucket={bucket_name}, "
            f"prefix={s3_prefix}, workers={max_workers}"
        )

    def _get_s3_key(self, local_path: str, base_dir: str) -> str:
        """Generate S3 key from local path."""
        # Get relative path from base directory
        rel_path = os.path.relpath(local_path, base_dir)
        return f"{self.s3_prefix}/{rel_path}" if self.s3_prefix else rel_path

    def _upload_file(
        self,
        local_path: str,
        s3_key: str,
        check_exists: bool = True,
    ) -> UploadResult:
        """Upload a single file to S3."""
        try:
            # Check if file exists locally
            if not os.path.exists(local_path):
                return UploadResult(
                    local_path=local_path,
                    s3_key=s3_key,
                    success=False,
                    error="File not found locally",
                )

            # Skip if already exists in S3
            if check_exists and self.skip_existing:
                if self.s3_manager.object_exists(s3_key):
                    return UploadResult(
                        local_path=local_path,
                        s3_key=s3_key,
                        success=True,
                        error="skipped",
                    )

            # Upload the file
            success = self.s3_manager.upload_file(local_path, s3_key)

            return UploadResult(
                local_path=local_path,
                s3_key=s3_key,
                success=success,
                error=None if success else "Upload failed",
            )

        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return UploadResult(
                local_path=local_path,
                s3_key=s3_key,
                success=False,
                error=str(e),
            )

    def _collect_files_from_video_log(self, video_log: OpenWorldVideoLog) -> list[str]:
        """Collect all unique frame paths from a video log."""
        all_paths: set[str] = set()

        for video in video_log.video_logs:
            for frame_path in video.video_log_paths:
                all_paths.add(frame_path)
                # Also add the corresponding metadata JSON
                json_path = frame_path.replace(".png", ".json")
                all_paths.add(json_path)

        return list(all_paths)

    def _load_video_log(self, json_path: str) -> Optional[OpenWorldVideoLog]:
        """Load an OpenWorldVideoLog from a JSON file."""
        try:
            with open(json_path, "r") as f:
                return OpenWorldVideoLog.model_validate_json(f.read())
        except Exception as e:
            logger.error(f"Error loading video log from {json_path}: {e}")
            return None

    def sync_from_json(
        self,
        json_path: str,
        base_dir: Optional[str] = None,
    ) -> SyncStats:
        """
        Sync frames from a single OpenWorldVideoLog JSON file to S3.

        Args:
            json_path: Path to the OpenWorldVideoLog JSON file
            base_dir: Base directory for computing relative S3 paths.
                      If None, uses the parent directory of json_path.

        Returns:
            SyncStats with upload statistics
        """
        video_log = self._load_video_log(json_path)
        if video_log is None:
            return SyncStats()

        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(json_path))

        return self._sync_video_log(video_log, base_dir)

    def sync_from_directory(
        self,
        directory: str,
        pattern: str = "*.json",
    ) -> SyncStats:
        """
        Sync frames from all OpenWorldVideoLog JSON files in a directory.

        Args:
            directory: Directory containing OpenWorldVideoLog JSON files
            pattern: Glob pattern for JSON files

        Returns:
            Combined SyncStats for all files
        """
        directory = os.path.abspath(directory)
        json_files = list(Path(directory).glob(pattern))

        if not json_files:
            logger.warning(f"No JSON files found in {directory} matching {pattern}")
            return SyncStats()

        logger.info(f"Found {len(json_files)} JSON files to process")

        combined_stats = SyncStats()

        for json_file in tqdm(json_files, desc="Processing JSON files"):
            video_log = self._load_video_log(str(json_file))
            if video_log is None:
                continue

            stats = self._sync_video_log(video_log, directory)
            combined_stats.total_files += stats.total_files
            combined_stats.uploaded += stats.uploaded
            combined_stats.skipped += stats.skipped
            combined_stats.failed += stats.failed

        return combined_stats

    def _sync_video_log(
        self,
        video_log: OpenWorldVideoLog,
        base_dir: str,
    ) -> SyncStats:
        """
        Sync all frames from a video log to S3.

        Args:
            video_log: The OpenWorldVideoLog to sync
            base_dir: Base directory for computing relative S3 paths

        Returns:
            SyncStats with upload statistics
        """
        # Collect all file paths
        all_paths = self._collect_files_from_video_log(video_log)
        stats = SyncStats(total_files=len(all_paths))

        if not all_paths:
            logger.warning("No files to sync")
            return stats

        logger.info(f"Syncing {len(all_paths)} files using {self.max_workers} threads")

        # Prepare upload tasks
        upload_tasks: list[tuple[str, str]] = []
        for local_path in all_paths:
            s3_key = self._get_s3_key(local_path, base_dir)
            upload_tasks.append((local_path, s3_key))

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._upload_file, local_path, s3_key): (
                    local_path,
                    s3_key,
                )
                for local_path, s3_key in upload_tasks
            }

            with tqdm(total=len(futures), desc="Uploading files") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result.success:
                        if result.error == "skipped":
                            stats.skipped += 1
                        else:
                            stats.uploaded += 1
                    else:
                        stats.failed += 1
                        logger.warning(
                            f"Failed to upload {result.local_path}: {result.error}"
                        )

                    pbar.update(1)

        logger.info(f"Sync complete: {stats}")
        return stats
