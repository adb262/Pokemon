#!/usr/bin/env python3
"""
S3 Utilities for Pokemon Training
Handles all S3 operations including file uploads, downloads, and listing.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class S3Manager:
    """Manages S3 operations for the Pokemon training pipeline"""

    def __init__(self, bucket_name: str, region_name: str = 'us-east-1',
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize S3 manager

        Args:
            bucket_name: S3 bucket name
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, can use env vars)
            aws_secret_access_key: AWS secret key (optional, can use env vars)
        """
        self.bucket_name = bucket_name
        self.region_name = region_name

        # Initialize S3 client
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })

        try:
            self.session = boto3.Session(**session_kwargs)
            self.s3_client = self.session.client('s3')
            self.s3_resource = self.session.resource('s3')
            self.bucket = self.s3_resource.Bucket(bucket_name)

            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")

        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or configure AWS CLI.")
            raise
        except ClientError as e:
            logger.error(f"Error connecting to S3 bucket {bucket_name}: {e}")
            raise

    def upload_file(self, local_path: Union[str, Path], s3_key: str,
                    extra_args: Optional[Dict] = None) -> bool:
        """
        Upload a file to S3

        Args:
            local_path: Local file path
            s3_key: S3 object key
            extra_args: Extra arguments for upload

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(
                str(local_path), self.bucket_name, s3_key,
                ExtraArgs=extra_args or {}
            )
            logger.debug(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading {local_path} to S3: {e}")
            return False

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> bool:
        """
        Download a file from S3

        Args:
            s3_key: S3 object key
            local_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.debug(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {s3_key} from S3: {e}")
            return False

    def upload_bytes(self, data: bytes, s3_key: str,
                     content_type: Optional[str] = None) -> bool:
        """
        Upload bytes data to S3

        Args:
            data: Bytes data to upload
            s3_key: S3 object key
            content_type: Content type for the object

        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                **extra_args
            )
            logger.debug(f"Uploaded bytes to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading bytes to S3: {e}")
            return False

    def download_bytes(self, s3_key: str) -> Optional[bytes]:
        """
        Download bytes data from S3

        Args:
            s3_key: S3 object key

        Returns:
            Bytes data if successful, None otherwise
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Error downloading bytes from S3: {e}")
            return None

    def list_objects(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List objects in S3 bucket with optional prefix and suffix filtering

        Args:
            prefix: Object key prefix to filter by
            suffix: Object key suffix to filter by

        Returns:
            List of object keys
        """
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if not suffix or key.endswith(suffix):
                            objects.append(key)

            return objects
        except Exception as e:
            logger.error(f"Error listing objects in S3: {e}")
            return []

    def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3

        Args:
            s3_key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3

        Args:
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.debug(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {s3_key} from S3: {e}")
            return False

    def upload_json(self, data: Dict[str, Any], s3_key: str) -> bool:
        """
        Upload JSON data to S3

        Args:
            data: Dictionary to upload as JSON
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            return self.upload_bytes(json_bytes, s3_key, 'application/json')
        except Exception as e:
            logger.error(f"Error uploading JSON to S3: {e}")
            return False

    def download_json(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Download JSON data from S3

        Args:
            s3_key: S3 object key

        Returns:
            Dictionary if successful, None otherwise
        """
        try:
            json_bytes = self.download_bytes(s3_key)
            if json_bytes:
                return json.loads(json_bytes.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error downloading JSON from S3: {e}")
            return None

    def upload_pytorch_model(self, model_state_dict: Dict[str, Any], s3_key: str) -> bool:
        """
        Upload PyTorch model state dict to S3

        Args:
            model_state_dict: Model state dictionary
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to bytes buffer
            buffer = io.BytesIO()
            torch.save(model_state_dict, buffer)
            buffer.seek(0)

            return self.upload_bytes(buffer.getvalue(), s3_key, 'application/octet-stream')
        except Exception as e:
            logger.error(f"Error uploading PyTorch model to S3: {e}")
            return False

    def download_pytorch_model(self, s3_key: str, map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
        """
        Download PyTorch model state dict from S3

        Args:
            s3_key: S3 object key
            map_location: Device to map tensors to

        Returns:
            Model state dictionary if successful, None otherwise
        """
        try:
            model_bytes = self.download_bytes(s3_key)
            if model_bytes:
                buffer = io.BytesIO(model_bytes)
                return torch.load(buffer, map_location=map_location)
            return None
        except Exception as e:
            logger.error(f"Error downloading PyTorch model from S3: {e}")
            return None

    def upload_image(self, image: Image.Image, s3_key: str, format: str = 'PNG') -> bool:
        """
        Upload PIL Image to S3

        Args:
            image: PIL Image object
            s3_key: S3 object key
            format: Image format (PNG, JPEG, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            buffer.seek(0)

            content_type = f'image/{format.lower()}'
            return self.upload_bytes(buffer.getvalue(), s3_key, content_type)
        except Exception as e:
            logger.error(f"Error uploading image to S3: {e}")
            return False

    def download_image(self, s3_key: str) -> Optional[Image.Image]:
        """
        Download PIL Image from S3

        Args:
            s3_key: S3 object key

        Returns:
            PIL Image if successful, None otherwise
        """
        try:
            image_bytes = self.download_bytes(s3_key)
            if image_bytes:
                buffer = io.BytesIO(image_bytes)
                return Image.open(buffer)
            return None
        except Exception as e:
            logger.error(f"Error downloading image from S3: {e}")
            return None

    def get_object_info(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata from S3

        Args:
            s3_key: S3 object key

        Returns:
            Object metadata if successful, None otherwise
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag')
            }
        except Exception as e:
            logger.error(f"Error getting object info from S3: {e}")
            return None


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse S3 path into bucket and key

    Args:
        s3_path: S3 path in format s3://bucket/key or bucket/key

    Returns:
        Tuple of (bucket_name, key)
    """
    if s3_path.startswith('s3://'):
        parsed = urlparse(s3_path)
        return parsed.netloc, parsed.path.lstrip('/')
    else:
        # Assume format is bucket/key
        parts = s3_path.split('/', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            raise ValueError(f"Invalid S3 path format: {s3_path}")


def get_s3_manager_from_env() -> S3Manager:
    """
    Create S3Manager from environment variables

    Environment variables:
    - S3_BUCKET_NAME: S3 bucket name
    - AWS_REGION: AWS region (default: us-east-1)
    - AWS_ACCESS_KEY_ID: AWS access key
    - AWS_SECRET_ACCESS_KEY: AWS secret key

    Returns:
        S3Manager instance
    """
    bucket_name = os.getenv('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is required")

    region_name = os.getenv('AWS_REGION', 'us-east-1')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")

    return S3Manager(
        bucket_name=bucket_name,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
