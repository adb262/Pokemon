#!/usr/bin/env python3
"""
S3 Upload Worker
Background worker for async S3 uploads using a thread pool.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from data.s3.s3_utils import default_s3_manager

logger = logging.getLogger(__name__)


class S3UploadWorker:
    """Background worker for async S3 uploads using a thread pool."""

    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: List = []
        self._started = False

    def start(self):
        """Start the upload worker thread pool."""
        if self._started:
            return

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures = []
        self._started = True
        logger.debug(f"Started S3 upload worker with {self.max_workers} threads")

    def submit_frame_upload(
        self,
        local_path: str,
        s3_key: str,
        delete_after_upload: bool = False,
    ):
        """Submit a frame for async upload."""
        if not self._started or self.executor is None:
            return

        future = self.executor.submit(
            self._upload_frame, local_path, s3_key, delete_after_upload
        )
        self.futures.append(future)

    def submit_metadata_upload(self, metadata_dict: Dict[str, Any], s3_key: str):
        """Submit metadata for async upload."""
        if not self._started or self.executor is None:
            return

        future = self.executor.submit(self._upload_metadata, metadata_dict, s3_key)
        self.futures.append(future)

    def _upload_frame(
        self,
        local_path: str,
        s3_key: str,
        delete_after_upload: bool,
    ) -> bool:
        """Upload a frame to S3."""
        try:
            success = default_s3_manager.upload_file(local_path, s3_key)

            if success:
                logger.debug(f"Uploaded frame to S3: {s3_key}")
                if delete_after_upload and os.path.exists(local_path):
                    try:
                        os.unlink(local_path)
                    except Exception as e:
                        logger.warning(f"Error removing local frame {local_path}: {e}")
                return True

            logger.warning(f"Failed to upload frame to S3: {s3_key}")
            return False

        except Exception as e:
            logger.error(f"Error uploading frame to S3: {e}")
            return False

    def _upload_metadata(self, metadata_dict: Dict[str, Any], s3_key: str) -> bool:
        """Upload metadata to S3."""
        try:
            success = default_s3_manager.upload_json(metadata_dict, s3_key)

            if success:
                logger.debug(f"Uploaded metadata to S3: {s3_key}")
                return True

            logger.warning(f"Failed to upload metadata to S3: {s3_key}")
            return False

        except Exception as e:
            logger.error(f"Error uploading metadata to S3: {e}")
            return False

    def wait_for_completion(self, timeout: Optional[float] = None) -> Tuple[int, int]:
        """
        Wait for all pending uploads to complete.

        Returns:
            Tuple of (success_count, failure_count)
        """
        if not self._started or not self.futures:
            return 0, 0

        success_count = 0
        failure_count = 0

        for future in as_completed(self.futures, timeout=timeout):
            try:
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                logger.error(f"Upload task failed with exception: {e}")
                failure_count += 1

        self.futures = []
        return success_count, failure_count

    def stop(self):
        """Stop the upload worker and wait for pending uploads."""
        if not self._started:
            return

        success, failed = self.wait_for_completion()
        logger.info(f"S3 uploads completed: {success} successful, {failed} failed")

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        self._started = False
        logger.debug("Stopped S3 upload worker")
