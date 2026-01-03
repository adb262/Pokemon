import argparse
import logging
import os
import sys

from src.data.s3.sync_dataset_to_s3 import DatasetS3Syncer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sync OpenWorldVideoLog dataset frames to S3"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to OpenWorldVideoLog JSON file or directory containing JSON files",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="S3 key prefix for uploads (default: empty)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files when source is a directory (default: *.json)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for computing relative S3 paths (default: source directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent upload threads (default: 32)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-upload files even if they exist in S3",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate source path
    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        logger.error(f"Source path does not exist: {source}")
        sys.exit(1)

    # Initialize syncer
    syncer = DatasetS3Syncer(
        bucket_name=args.bucket,
        s3_prefix=args.prefix,
        max_workers=args.workers,
        skip_existing=not args.no_skip_existing,
        region_name=args.region,
    )

    # Run sync
    if os.path.isfile(source):
        logger.info(f"Syncing single file: {source}")
        stats = syncer.sync_from_json(source, base_dir=args.base_dir)
    else:
        logger.info(f"Syncing directory: {source}")
        stats = syncer.sync_from_directory(source, pattern=args.pattern)

    # Print final stats
    print(f"\n{'=' * 50}")
    print("Sync Complete!")
    print(f"{'=' * 50}")
    print(f"Total files:    {stats.total_files}")
    print(f"Uploaded:       {stats.uploaded}")
    print(f"Skipped:        {stats.skipped}")
    print(f"Failed:         {stats.failed}")
    print(f"{'=' * 50}")

    if stats.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
