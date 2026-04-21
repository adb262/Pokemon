import logging
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
import traceback
from typing import Optional

import torch
import torchvision.utils as vutils
import tyro

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from dynamics_model.create_model import create_dynamics_model
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.create_model import create_action_model_from_dynamics_config
from video_tokenization.checkpoints import load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_CONFIG = DynamicsModelTrainingConfig()


@dataclass
class InteractiveInferenceArgs:
    tokenizer_checkpoint_path: str
    dynamics_model_checkpoint_path: str
    device: str = BASE_CONFIG.device
    num_images_in_video: int = BASE_CONFIG.num_images_in_video
    image_size: int = BASE_CONFIG.image_size
    action_bins: list[int] = field(default_factory=lambda: BASE_CONFIG.action_bins.copy())
    action_d_model: int = BASE_CONFIG.action_d_model
    action_num_transformer_layers: int = BASE_CONFIG.action_num_transformer_layers
    action_num_heads: int = BASE_CONFIG.action_num_heads
    action_latent_dim: int = BASE_CONFIG.action_latent_dim
    dynamics_d_model: int = BASE_CONFIG.dynamics_d_model
    dynamics_num_transformer_layers: int = BASE_CONFIG.dynamics_num_transformer_layers
    dynamics_num_heads: int = BASE_CONFIG.dynamics_num_heads
    batch_size: int = BASE_CONFIG.batch_size
    frames_dir: str = BASE_CONFIG.frames_dir
    local_cache_dir: Optional[str] = BASE_CONFIG.local_cache_dir
    use_s3: bool = BASE_CONFIG.use_s3
    dataset_train_key: Optional[str] = BASE_CONFIG.dataset_train_key
    sync_from_s3: bool = BASE_CONFIG.sync_from_s3
    output_dir: str = "dynamics_model_results/inference"


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def _build_config(args: InteractiveInferenceArgs) -> DynamicsModelTrainingConfig:
    return replace(
        BASE_CONFIG,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=args.dynamics_model_checkpoint_path,
        device=args.device,
        num_images_in_video=args.num_images_in_video,
        image_size=args.image_size,
        action_bins=args.action_bins,
        action_d_model=args.action_d_model,
        action_num_transformer_layers=args.action_num_transformer_layers,
        action_num_heads=args.action_num_heads,
        action_latent_dim=args.action_latent_dim,
        dynamics_d_model=args.dynamics_d_model,
        dynamics_num_transformer_layers=args.dynamics_num_transformer_layers,
        dynamics_num_heads=args.dynamics_num_heads,
        use_wandb=False,
        batch_size=args.batch_size,
        frames_dir=args.frames_dir,
        local_cache_dir=args.local_cache_dir,
        use_s3=args.use_s3,
        dataset_train_key=args.dataset_train_key,
        sync_from_s3=args.sync_from_s3,
    )


def _load_dynamics_model(
    config: DynamicsModelTrainingConfig, device: torch.device
):
    tokenizer, _ = load_model_from_checkpoint(config.tokenizer_checkpoint_path, device)
    _freeze_module(tokenizer)

    action_model = create_action_model_from_dynamics_config(config).to(device)
    action_model.eval()

    dynamics_model = create_dynamics_model(config, tokenizer, action_model).to(device)

    if config.dynamics_model_checkpoint_path:
        checkpoint = torch.load(config.dynamics_model_checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        dynamics_model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded dynamics model weights from %s", config.dynamics_model_checkpoint_path)

    dynamics_model.eval()
    return dynamics_model


def _build_test_dataloader(
    config: DynamicsModelTrainingConfig,
) -> VideoWindowLoader:
    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required for inference dataloader")

    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )

    _train_dataset, test_dataset = build_datasets(config, local_cache)

    test_dataloader = VideoWindowLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=4,
        seed=config.seed,
    )

    logger.info("Test dataloader ready with %d videos", len(test_dataset))
    return test_dataloader


def _normalize_frame(frame: torch.Tensor) -> torch.Tensor:
    """Normalize a frame tensor to 0-1 range."""
    frame = frame.detach()
    return (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)


def _save_concatenated_frames(
    frames: torch.Tensor,
    output_dir: Path,
    prefix: str,
    step: int,
) -> str:
    """Save all frames as a single horizontal concatenation.
    
    Args:
        frames: (T, C, H, W) or (1, T, C, H, W) tensor
        output_dir: Directory to save to
        prefix: Filename prefix
        step: Step number for filename
    
    Returns:
        Path to saved image
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if frames.dim() == 5:
        frames = frames[0]  # (T, C, H, W)
    
    # Normalize each frame independently
    normalized = torch.stack([_normalize_frame(f) for f in frames], dim=0)
    
    # Save as horizontal grid (nrow = number of frames for single row)
    path = output_dir / f"test_step.png"
    vutils.save_image(normalized.cpu(), path, nrow=normalized.size(0), padding=2)
    return str(path)


def _interactive_loop(
    model,
    test_dataloader: VideoWindowLoader,
    config: DynamicsModelTrainingConfig,
    args: InteractiveInferenceArgs,
    device: torch.device,
) -> None:
    logger.info("Interactive inference ready.")
    logger.info("Commands: <action_id> (int) to generate, 'new' for new video, 'q' to quit.")

    loader_iter = iter(test_dataloader)

    def _sample_new_video() -> torch.Tensor:
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_dataloader)
            batch = next(loader_iter)

        if batch.dim() != 5:
            raise ValueError("Expected batch shape (B, T, C, H, W)")
        idx = random.randint(0, batch.size(0) - 1)
        video = batch[idx : idx + 1].to(device)
        logger.info("Selected random video idx=%d from batch size=%d", idx, batch.size(0))
        return video

    # Current sliding window of frames: (1, T, C, H, W)
    current_video: Optional[torch.Tensor] = None
    step_counter = 0
    session_id = 0
    output_dir = Path(args.output_dir)
    total_video: Optional[torch.Tensor] = None

    while True:
        # If no current video, sample one
        if current_video is None:
            try:
                current_video = _sample_new_video()
                current_video = current_video[:, :1, :, :, :]
                total_video = current_video.clone()
                session_id += 1
                step_counter = 0
                prefix = f"session_{session_id:03d}"
                path = _save_concatenated_frames(current_video, output_dir, prefix, step_counter)
                print(f"[session {session_id}] Initial frames saved to: {path}")
            except Exception as exc:
                logger.error("Failed to sample initial video: %s", exc)
                continue

        cmd = input("Action id (int), 'new', or 'q': ").strip()
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if cmd.lower() == "new":
            current_video = None
            print("Switching to new video...")
            continue

        try:
            action_id = int(cmd)
        except ValueError:
            print("Please provide a valid integer for action id, 'new', or 'q'.")
            continue

        with torch.no_grad():
            try:
                action_tensor = torch.tensor(
                    [action_id], dtype=torch.long, device=device
                )
                # inference returns the predicted next frame(s)
                result = model.inference(current_video, action_tensor)
            except NotImplementedError:
                logger.warning(
                    "DynamicsModel.inference is a stub and must be implemented before use."
                )
                continue
            except Exception as exc:
                traceback.print_exc()
                logger.error("Inference call failed: %s", exc)
                continue

        # result should be shape (1, 1, C, H, W) or (1, C, H, W) for the new frame
        # Update sliding window: drop oldest frame, append new frame
        if result.dim() == 4:
            new_frame = result.unsqueeze(1)  # (1, 1, C, H, W)
        elif result.dim() == 5:
            new_frame = result[:, -1:, :, :, :]  # take last generated frame
        else:
            new_frame = result.unsqueeze(0).unsqueeze(0)

        # Slide window: keep last (num_frames - 1) frames, append new frame
        current_video = torch.cat([current_video[:, 1:, :, :, :], new_frame], dim=1)

        if total_video is None:
            raise ValueError("total_video is None")

        total_video = torch.cat([total_video, new_frame], dim=1)

        step_counter += 1
        prefix = f"session_{session_id:03d}"
        path = _save_concatenated_frames(total_video, output_dir, prefix, step_counter)
        print(f"[session {session_id}, step {step_counter}] action={action_id} saved to: {path}")


def main():
    args = tyro.cli(InteractiveInferenceArgs)
    device = torch.device(args.device)
    config = _build_config(args)
    model = _load_dynamics_model(config, device)
    test_dataloader = _build_test_dataloader(config)
    _interactive_loop(model, test_dataloader, config, args, device)


if __name__ == "__main__":
    main()

