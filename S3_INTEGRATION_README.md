# S3 Integration for Pokemon VQVAE Training

This document explains how to use the S3 integration for the Pokemon VQVAE training pipeline. The S3 integration allows you to:

- Store training data (Pokemon frames) in S3
- Save model checkpoints to S3
- Upload training logs and TensorBoard data to S3
- Cache frequently accessed images locally for performance

## Prerequisites

1. **AWS Account**: You need an AWS account with S3 access
2. **S3 Bucket**: Create an S3 bucket for storing your training data and outputs
3. **AWS Credentials**: Configure AWS credentials (see setup section below)
4. **Dependencies**: Ensure `boto3` is installed (already included in `pyproject.toml`)

## Setup

### 1. AWS Credentials

Set up your AWS credentials using one of these methods:

#### Option A: Environment Variables (Recommended)
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export S3_BUCKET_NAME=your-pokemon-training-bucket
export AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

#### Option B: AWS CLI Configuration
```bash
aws configure
```

#### Option C: IAM Roles (for EC2 instances)
If running on EC2, attach an IAM role with S3 permissions to your instance.

### 2. S3 Bucket Structure

Organize your S3 bucket with the following structure:
```
your-bucket-name/
├── pokemon_frames/           # Training data
│   ├── game1/
│   │   ├── episode1/
│   │   │   ├── frame_001.png
│   │   │   ├── frame_002.png
│   │   │   └── ...
│   │   └── episode2/
│   └── game2/
├── checkpoints/              # Model checkpoints (auto-created)
│   └── experiment_name/
├── tensorboard/              # TensorBoard logs (auto-created)
│   └── experiment_name/
└── logs/                     # Training logs (auto-created)
    └── experiment_name/
```

### 3. Upload Training Data

Upload your Pokemon frames to S3. You can use the AWS CLI:

```bash
# Upload frames directory to S3
aws s3 sync ./pokemon_frames s3://your-bucket-name/pokemon_frames/

# Or upload with specific options
aws s3 sync ./pokemon_frames s3://your-bucket-name/pokemon_frames/ \
    --storage-class STANDARD_IA \
    --exclude "*.DS_Store"
```

## Usage

### Method 1: Using the Example Script

The easiest way to get started is using the provided example script:

```bash
python example_s3_training.py
```

This script will:
1. Check your environment variables
2. Configure training for S3
3. Show the configuration
4. Ask for confirmation before starting

### Method 2: Command Line Arguments

You can use the training script directly with S3 options:

```bash
python idm/train.py \
    --use-s3 \
    --s3-bucket your-bucket-name \
    --s3-region us-east-1 \
    --frames-dir pokemon_frames \
    --cache-dir /tmp/pokemon_cache \
    --experiment-name my_s3_experiment \
    --batch-size 16 \
    --epochs 50
```

### Method 3: Programmatic Configuration

You can also configure S3 settings programmatically:

```python
from train import CONFIG, main

# Configure S3 settings
CONFIG.use_s3 = True
CONFIG.s3_bucket = 'your-bucket-name'
CONFIG.s3_region = 'us-east-1'
CONFIG.frames_dir = 'pokemon_frames'
CONFIG.local_cache_dir = '/tmp/pokemon_cache'
CONFIG.experiment_name = 'my_experiment'

# Start training
main()
```

## Configuration Options

### S3-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `use_s3` | Enable S3 integration | `False` |
| `s3_bucket` | S3 bucket name | `None` (uses env var) |
| `s3_region` | AWS region | `us-east-1` |
| `s3_checkpoint_prefix` | S3 prefix for checkpoints | `checkpoints` |
| `s3_tensorboard_prefix` | S3 prefix for TensorBoard logs | `tensorboard` |
| `s3_logs_prefix` | S3 prefix for training logs | `logs` |
| `local_cache_dir` | Local cache directory | `None` (uses temp dir) |
| `max_cache_size` | Max images to cache locally | `1000` |

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `frames_dir` | S3 prefix for training frames | `pokemon_frames` |
| `experiment_name` | Name for this experiment | Auto-generated |
| `batch_size` | Training batch size | `16` |
| `num_epochs` | Number of training epochs | `50` |
| `learning_rate` | Learning rate | `1e-4` |

## Features

### 1. Automatic Caching

The system automatically caches frequently accessed images locally to improve performance:
- Images are downloaded from S3 on first access
- Cached locally for subsequent uses
- LRU eviction when cache is full
- Configurable cache size

### 2. Resumable Training

Training can be resumed from S3 checkpoints:
```bash
python idm/train.py \
    --use-s3 \
    --resume s3://your-bucket/checkpoints/experiment/checkpoint_latest.pt
```

### 3. Real-time Uploads

- Checkpoints are uploaded to S3 during training
- Logs are uploaded periodically
- TensorBoard data is uploaded at the end of each epoch

### 4. Fallback to Local Storage

If S3 setup fails, the system automatically falls back to local storage with a warning.

## Monitoring and Debugging

### Check S3 Uploads

Monitor your S3 bucket to see uploads in real-time:
```bash
# List recent checkpoints
aws s3 ls s3://your-bucket-name/checkpoints/experiment_name/ --recursive

# List logs
aws s3 ls s3://your-bucket-name/logs/experiment_name/ --recursive

# Download a specific checkpoint
aws s3 cp s3://your-bucket-name/checkpoints/experiment_name/checkpoint_best.pt ./
```

### View Logs

Training logs include S3 operation status:
```bash
# View local logs (if any)
tail -f training.log

# Download and view S3 logs
aws s3 cp s3://your-bucket-name/logs/experiment_name/training_latest.log ./
tail -f training_latest.log
```

### TensorBoard with S3

To view TensorBoard logs stored in S3:
```bash
# Download TensorBoard data
aws s3 sync s3://your-bucket-name/tensorboard/experiment_name/ ./tensorboard_data/

# Start TensorBoard
tensorboard --logdir ./tensorboard_data/
```

## Performance Considerations

### 1. Network Bandwidth

- S3 operations require network bandwidth
- Consider using larger batch sizes to amortize S3 overhead
- Use local caching effectively

### 2. S3 Costs

- Standard storage for frequently accessed data
- Consider IA (Infrequent Access) for archival
- Use lifecycle policies for old checkpoints

### 3. Cache Management

- Larger cache sizes reduce S3 requests but use more local storage
- Monitor local disk space when using large caches
- Cache is automatically cleaned up on exit

## Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   ```
   Error: AWS credentials not found
   ```
   Solution: Set environment variables or configure AWS CLI

2. **S3 Bucket Access Denied**
   ```
   Error: Access denied to S3 bucket
   ```
   Solution: Check bucket permissions and AWS credentials

3. **Network Connectivity Issues**
   ```
   Error: Unable to connect to S3
   ```
   Solution: Check internet connection and AWS region

4. **Cache Directory Issues**
   ```
   Error: Cannot create cache directory
   ```
   Solution: Ensure write permissions for cache directory

### Debug Mode

Enable debug logging for S3 operations:
```python
import logging
logging.getLogger('boto3').setLevel(logging.DEBUG)
logging.getLogger('botocore').setLevel(logging.DEBUG)
```

## Security Best Practices

1. **Use IAM Roles**: Prefer IAM roles over access keys when possible
2. **Least Privilege**: Grant minimal required S3 permissions
3. **Bucket Policies**: Use bucket policies to restrict access
4. **Encryption**: Enable S3 server-side encryption
5. **Access Logging**: Enable S3 access logging for audit trails

## Example IAM Policy

Minimal IAM policy for S3 access:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## Support

For issues with S3 integration:
1. Check the troubleshooting section above
2. Review AWS CloudTrail logs for S3 API calls
3. Verify bucket permissions and policies
4. Test S3 access with AWS CLI first 