# from beartype import BeartypeConf
# from beartype.claw import beartype_all
from dataclasses import dataclass
from typing import Callable, Optional
from data_collection.pokemon_frame_loader import PokemonFrameLoader
from idm.loss_fns import reconstruction_loss
from idm.vqvae import VQVAE
from idm.s3_utils import S3Manager, get_s3_manager_from_env
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import tempfile

# Add the parent directory to the path so we can import from data_collection
sys.path.append(str(Path(__file__).parent.parent))


# beartype_all(conf=BeartypeConf(violation_type=UserWarning))
# AI BS generation here... use a real config class for god sake

@dataclass
class TrainingConfig:
    image_size: int = 400
    patch_size: int = 16
    batch_size: int = 16
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    num_epochs: int = 50
    device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    frames_dir: str = 'pokemon_frames'
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 1000
    checkpoint_dir: str = 'checkpoints'
    tensorboard_dir: str = 'runs'
    seed: int = 42
    hidden_dim: int = 256
    num_transformer_layers: int = 6
    latent_dim: int = 128
    num_embeddings: int = 8
    num_heads: int = 4
    resume_from: Optional[str] = None
    experiment_name: Optional[str] = None
    # S3 Configuration
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_region: str = 'us-east-1'
    s3_checkpoint_prefix: str = 'baseten_test_checkpoints'
    s3_tensorboard_prefix: str = 'baseten_test_tensorboard'
    s3_logs_prefix: str = 'baseten_test_logs'
    local_cache_dir: Optional[str] = None
    max_cache_size: int = 1000
    # Temporary attributes for S3 operations
    _temp_log_file: Optional[str] = None
    _temp_tensorboard_dir: Optional[str] = None


CONFIG = TrainingConfig()


def setup_logging(config: TrainingConfig, s3_manager: Optional[S3Manager] = None):
    """Setup logging configuration with S3 support"""
    log_handlers = []

    if config.use_s3 and s3_manager:
        # Create a temporary log file that will be uploaded to S3 periodically
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        log_handlers.append(logging.FileHandler(log_file.name))

        # Store the temp log file path for later S3 upload
        config._temp_log_file = log_file.name
    else:
        log_handlers.append(logging.FileHandler('training.log'))

    log_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )


def upload_logs_to_s3(config: TrainingConfig, s3_manager: S3Manager):
    """Upload logs to S3"""
    if config._temp_log_file and os.path.exists(config._temp_log_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_log_key = f"{config.s3_logs_prefix}/{config.experiment_name}/training_{timestamp}.log"

        success = s3_manager.upload_file(config._temp_log_file, s3_log_key)
        if success:
            logging.info(f"Uploaded logs to S3: {s3_log_key}")
        else:
            logging.error(f"Failed to upload logs to S3: {s3_log_key}")


def create_model(config: TrainingConfig):
    """Create and initialize the VQVAE model"""
    num_patches = (config.image_size // config.patch_size) ** 2

    model = VQVAE(
        channels=3,
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        patch_embed_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        num_heads=config.num_heads,
        num_patches=num_patches,
        num_transformer_layers=config.num_transformer_layers
    )

    return model


def save_checkpoint(
        model, optimizer, scheduler, epoch, batch_idx, loss, config, dataloader_state,
        checkpoint_dir, s3_manager: Optional[S3Manager] = None, is_best=False):
    """Save comprehensive model checkpoint to local storage or S3"""

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'dataloader_state': dataloader_state,
        'timestamp': datetime.now().isoformat(),
        'total_batches_processed': epoch * len(dataloader_state['loader_state']['dataset_state']) // config.batch_size + batch_idx
    }

    if config.use_s3 and s3_manager:
        # Save to S3
        checkpoint_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        latest_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_latest.pt"

        # Upload checkpoint
        success = s3_manager.upload_pytorch_model(checkpoint, checkpoint_key)
        if success:
            logging.info(f"Checkpoint saved to S3: {checkpoint_key}")

            # Also save as latest
            s3_manager.upload_pytorch_model(checkpoint, latest_key)

            # Save best checkpoint if this is the best
            if is_best:
                best_key = f"{config.s3_checkpoint_prefix}/{config.experiment_name}/checkpoint_best.pt"
                s3_manager.upload_pytorch_model(checkpoint, best_key)
                logging.info(f"New best checkpoint saved to S3: {best_key}")

            return checkpoint_key
        else:
            logging.error(f"Failed to save checkpoint to S3: {checkpoint_key}")
            return None
    else:
        # Save locally
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint if this is the best
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"New best checkpoint saved: {best_path}")

        logging.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device,
                    s3_manager: Optional[S3Manager] = None):
    """Load comprehensive model checkpoint from local storage or S3"""

    if checkpoint_path.startswith('s3://') or (s3_manager and not os.path.exists(checkpoint_path)):
        # Load from S3
        if s3_manager is None:
            logging.error("S3Manager required for S3 checkpoint loading")
            return None

        checkpoint = s3_manager.download_pytorch_model(checkpoint_path, map_location=device)
        if checkpoint is None:
            logging.error(f"Checkpoint not found in S3: {checkpoint_path}")
            return None
    else:
        # Load locally
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    batch_idx = checkpoint.get('batch_idx', 0)
    loss = checkpoint['loss']
    config = checkpoint.get('config', {})
    dataloader_state = checkpoint.get('dataloader_state', {})

    logging.info(f"Checkpoint loaded: {checkpoint_path}")
    logging.info(f"Resuming from epoch {epoch}, batch {batch_idx}, loss: {loss:.6f}")

    return {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'loss': loss,
        'config': config,
        'dataloader_state': dataloader_state
    }


def log_model_info(model, writer, step=0):
    """Log model information to tensorboard"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    writer.add_scalar('Model/TotalParameters', total_params, step)
    writer.add_scalar('Model/TrainableParameters', trainable_params, step)

    logging.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")


def evaluate_model(model: VQVAE, dataloader: PokemonFrameLoader,
                   criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                   device: torch.device, num_batches: int = 10):
    """Evaluate model on a subset of data"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (image1_batch, image2_batch) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            image1_batch = image1_batch.to(device)
            image2_batch = image2_batch.to(device)

            try:
                decoded = model(image1_batch, image2_batch)
                loss = criterion(image1_batch, image2_batch, decoded)

                total_loss += loss.item() * image1_batch.size(0)
                total_samples += image1_batch.size(0)

            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    model.train()  # Switch back to training mode
    return avg_loss


def train_epoch(
        model: VQVAE, dataloader: PokemonFrameLoader, optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.CosineAnnealingLR,
        criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                            torch.Tensor],
        device: torch.device, epoch: int, writer: SummaryWriter, config: TrainingConfig,
        s3_manager: Optional[S3Manager] = None, start_batch: int = 0):
    """Train for one epoch with comprehensive logging"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    best_loss = float('inf')

    # Set up resumable dataloader
    if start_batch > 0:
        dataloader.resumable_loader.set_epoch(epoch)
        dataloader.resumable_loader.current_batch = start_batch

    epoch_start_time = time.time()

    for batch_idx, (image1_batch, image2_batch) in enumerate(dataloader):
        # Skip batches if resuming
        if batch_idx < start_batch:
            continue

        batch_start_time = time.time()

        # Move to device
        image1_batch = image1_batch.to(device)
        image2_batch = image2_batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        try:
            decoded = model(image1_batch, image2_batch)

            # Calculate loss (reconstruction loss)
            loss = criterion(image1_batch, image2_batch, decoded)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Step scheduler every batch for cosine annealing

            total_loss += loss.item()
            batch_time = time.time() - batch_start_time

            # Calculate global step
            global_step = epoch * num_batches + batch_idx

            # Log to tensorboard
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Time/Batch_Time', batch_time, global_step)

            # Log progress
            if batch_idx % config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}, LR: {current_lr:.2e}, '
                    f'Time: {batch_time:.2f}s'
                )

            # Save checkpoint periodically
            if batch_idx > 0 and batch_idx % config.save_interval == 0:
                dataloader_state = dataloader.get_state()
                is_best = loss.item() < best_loss
                if is_best:
                    best_loss = loss.item()

                save_checkpoint(
                    model, optimizer, scheduler, epoch, batch_idx,
                    loss.item(), config, dataloader_state,
                    config.checkpoint_dir, s3_manager, is_best
                )

            # Evaluate periodically
            if batch_idx > 0 and batch_idx % config.eval_interval == 0:
                eval_loss = evaluate_model(model, dataloader, criterion, device)
                writer.add_scalar('Loss/Eval', eval_loss, global_step)
                logging.info(f'Evaluation loss at batch {batch_idx}: {eval_loss:.6f}')

            # Upload logs to S3 periodically
            if config.use_s3 and s3_manager and batch_idx > 0 and batch_idx % (config.save_interval * 2) == 0:
                upload_logs_to_s3(config, s3_manager)

        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time

    # Log epoch metrics
    writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    writer.add_scalar('Time/Epoch_Time', epoch_time, epoch)

    logging.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s')
    return avg_loss


def setup_s3_manager(config: TrainingConfig) -> Optional[S3Manager]:
    """Setup S3 manager if S3 is enabled"""
    if not config.use_s3:
        return None

    try:
        if config.s3_bucket:
            # Use explicit bucket configuration
            s3_manager = S3Manager(
                bucket_name=config.s3_bucket,
                region_name=config.s3_region
            )
        else:
            # Use environment variables
            s3_manager = get_s3_manager_from_env()

        logging.info(f"S3 manager initialized for bucket: {s3_manager.bucket_name}")
        return s3_manager

    except Exception as e:
        logging.error(f"Failed to initialize S3 manager: {e}")
        logging.error("Falling back to local storage")
        config.use_s3 = False
        return None


def create_tensorboard_writer(config: TrainingConfig, s3_manager: Optional[S3Manager] = None):
    """Create tensorboard writer with S3 support"""
    if config.use_s3 and s3_manager:
        # Create a temporary directory for tensorboard logs
        temp_dir = tempfile.mkdtemp(prefix="tensorboard_")
        if config.experiment_name:
            tensorboard_path = os.path.join(temp_dir, config.experiment_name)
        else:
            tensorboard_path = temp_dir

        # Store temp directory for later S3 upload
        config._temp_tensorboard_dir = temp_dir
    else:
        if config.experiment_name:
            tensorboard_path = os.path.join(config.tensorboard_dir, config.experiment_name)
        else:
            tensorboard_path = config.tensorboard_dir

    return SummaryWriter(tensorboard_path)


def upload_tensorboard_to_s3(config: TrainingConfig, s3_manager: S3Manager):
    """Upload tensorboard logs to S3"""
    if config._temp_tensorboard_dir and os.path.exists(config._temp_tensorboard_dir):
        # Upload all files in the tensorboard directory
        for root, dirs, files in os.walk(config._temp_tensorboard_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, config._temp_tensorboard_dir)
                s3_key = f"{config.s3_tensorboard_prefix}/{relative_path}"

                success = s3_manager.upload_file(local_path, s3_key)
                if success:
                    logging.debug(f"Uploaded tensorboard file to S3: {s3_key}")
                else:
                    logging.warning(f"Failed to upload tensorboard file to S3: {s3_key}")


def main():
    """Main training function with resumable training support and S3 integration"""

    # Setup S3 manager first
    s3_manager = setup_s3_manager(CONFIG)

    setup_logging(CONFIG, s3_manager)

    # Generate experiment name if not provided
    if CONFIG.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        CONFIG.experiment_name = f"pokemon_vqvae_{timestamp}"

    logging.info(f"Starting Pokemon VQVAE training - Experiment: {CONFIG.experiment_name}")
    logging.info(f"Using S3: {CONFIG.use_s3}")
    if CONFIG.use_s3 and s3_manager:
        logging.info(f"S3 Bucket: {s3_manager.bucket_name}")

    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG.seed)
    torch.cuda.manual_seed_all(CONFIG.seed)

    # Create device
    device = torch.device(CONFIG.device)
    logging.info(f"Using device: {device}")

    # Create data loader
    logging.info("Creating data loader...")
    dataloader = PokemonFrameLoader(
        frames_dir=CONFIG.frames_dir,
        batch_size=CONFIG.batch_size,
        image_size=CONFIG.image_size,
        shuffle=True,
        num_workers=2,
        min_frame_gap=1,
        max_frame_gap=3,
        seed=CONFIG.seed,
        use_s3=CONFIG.use_s3,
        s3_manager=s3_manager,
        cache_dir=CONFIG.local_cache_dir,
        max_cache_size=CONFIG.max_cache_size
    )

    # Print dataset info
    info = dataloader.get_dataset_info()
    logging.info("Dataset Info:")
    for key, value in info.items():
        logging.info(f"  {key}: {value}")

    # Create model
    logging.info("Creating model...")
    model = create_model(CONFIG)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=1e-4)
    logging.info(f"Optimizer created with learning rate: {CONFIG.learning_rate}")

    # Cosine annealing scheduler
    total_steps = CONFIG.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=CONFIG.min_learning_rate
    )

    # Loss function
    criterion = reconstruction_loss

    # Create tensorboard writer
    writer = create_tensorboard_writer(CONFIG, s3_manager)

    # Log model info
    log_model_info(model, writer)

    # Log configuration
    config_text = json.dumps({**CONFIG.__dict__}, indent=2, default=str)
    writer.add_text('Config', config_text, 0)

    # Resume from checkpoint if specified
    start_epoch = 0
    start_batch = 0

    if CONFIG.resume_from:
        checkpoint_info = load_checkpoint(
            CONFIG.resume_from, model, optimizer, scheduler, device, s3_manager
        )
        if checkpoint_info:
            start_epoch = checkpoint_info['epoch']
            start_batch = checkpoint_info.get('batch_idx', 0)

            # Restore dataloader state
            if 'dataloader_state' in checkpoint_info:
                dataloader_state = checkpoint_info['dataloader_state']
                dataloader.resumable_loader = dataloader.create_resumable_loader(
                    start_epoch, start_batch
                )

    # Training loop
    logging.info("Starting training loop...")
    best_loss = float('inf')

    try:
        for epoch in range(start_epoch, CONFIG.num_epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0

            avg_loss = train_epoch(
                model, dataloader, optimizer, scheduler, criterion, device,
                epoch, writer, CONFIG, s3_manager, epoch_start_batch
            )

            # Save end-of-epoch checkpoint
            dataloader_state = dataloader.get_state()
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            save_checkpoint(
                model, optimizer, scheduler, epoch, len(dataloader),
                avg_loss, CONFIG, dataloader_state,
                CONFIG.checkpoint_dir, s3_manager, is_best
            )

            # Upload logs and tensorboard to S3 at end of epoch
            if CONFIG.use_s3 and s3_manager:
                upload_logs_to_s3(CONFIG, s3_manager)
                upload_tensorboard_to_s3(CONFIG, s3_manager)

            # Reset start_batch for subsequent epochs
            start_batch = 0

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Save checkpoint on interruption
        dataloader_state = dataloader.get_state()
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            dataloader.resumable_loader.current_batch,
            avg_loss, CONFIG, dataloader_state, CONFIG.checkpoint_dir, s3_manager
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        writer.close()

        # Final upload to S3
        if CONFIG.use_s3 and s3_manager:
            upload_logs_to_s3(CONFIG, s3_manager)
            upload_tensorboard_to_s3(CONFIG, s3_manager)

        # Cleanup
        dataloader.cleanup()

        # Cleanup temporary directories
        if CONFIG._temp_log_file and os.path.exists(CONFIG._temp_log_file):
            os.unlink(CONFIG._temp_log_file)
        if CONFIG._temp_tensorboard_dir and os.path.exists(CONFIG._temp_tensorboard_dir):
            import shutil
            shutil.rmtree(CONFIG._temp_tensorboard_dir)

    logging.info("Training completed!")
    logging.info(f"Best loss achieved: {best_loss:.6f}")

    if CONFIG.use_s3 and s3_manager:
        logging.info(f"Checkpoints and logs saved to S3 bucket: {s3_manager.bucket_name}")
    else:
        logging.info(
            f"Tensorboard logs saved to: {os.path.join(CONFIG.tensorboard_dir, CONFIG.experiment_name or 'default')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pokemon VQVAE Training')
    parser.add_argument('--frames-dir', type=str, help='Frames directory (local path or S3 prefix)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, help='Name for this experiment')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--use-s3', action='store_true', help='Use S3 for storage')
    parser.add_argument('--frames-dir', type=str, help='Frames directory (local path or S3 prefix)')
    parser.add_argument('--cache-dir', type=str, help='Local cache directory for S3 images')

    args = parser.parse_args()

    # Update config with command line arguments
    if args.resume:
        CONFIG.resume_from = args.resume
    if args.experiment_name:
        CONFIG.experiment_name = args.experiment_name
    if args.epochs:
        CONFIG.num_epochs = args.epochs
    if args.batch_size:
        CONFIG.batch_size = args.batch_size
    if args.lr:
        CONFIG.learning_rate = args.lr
    if args.frames_dir:
        CONFIG.frames_dir = args.frames_dir
    if args.use_s3:
        CONFIG.use_s3 = True
        CONFIG.s3_bucket = os.getenv('S3_BUCKET_NAME')
        CONFIG.s3_region = os.getenv('AWS_REGION', 'us-east-1')
    if args.frames_dir:
        CONFIG.frames_dir = args.frames_dir
    if args.cache_dir:
        CONFIG.local_cache_dir = args.cache_dir

    main()
