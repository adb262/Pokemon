import logging
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
import traceback
from typing import Literal, Optional

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


def _load_checkpoint_config(checkpoint_path: Optional[str]) -> dict:
    if checkpoint_path is None:
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return {}
    return checkpoint_config


@dataclass
class InteractiveInferenceArgs:
    tokenizer_checkpoint_path: str
    dynamics_model_checkpoint_path: str
    action_model_checkpoint_path: Optional[str] = BASE_CONFIG.action_model_checkpoint_path
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
    dataset_type: Literal["pokemon", "atari_pong"] = BASE_CONFIG.dataset_type
    atari_pong_data_dir: Optional[str] = BASE_CONFIG.atari_pong_data_dir
    frames_dir: str = BASE_CONFIG.frames_dir
    local_cache_dir: Optional[str] = BASE_CONFIG.local_cache_dir
    use_s3: bool = BASE_CONFIG.use_s3
    dataset_train_key: Optional[str] = BASE_CONFIG.dataset_train_key
    sync_from_s3: bool = BASE_CONFIG.sync_from_s3
    output_dir: str = "dynamics_model_results/inference"
    max_steps: int = BASE_CONFIG.rollout_max_steps


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def _build_config(args: InteractiveInferenceArgs) -> DynamicsModelTrainingConfig:
    config = replace(
        BASE_CONFIG,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=args.dynamics_model_checkpoint_path,
        action_model_checkpoint_path=args.action_model_checkpoint_path,
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
        dataset_type=args.dataset_type,
        atari_pong_data_dir=args.atari_pong_data_dir,
        frames_dir=args.frames_dir,
        local_cache_dir=args.local_cache_dir,
        use_s3=args.use_s3,
        dataset_train_key=args.dataset_train_key,
        sync_from_s3=args.sync_from_s3,
    )

    dynamics_checkpoint_config = _load_checkpoint_config(
        args.dynamics_model_checkpoint_path
    )
    action_checkpoint_config = _load_checkpoint_config(args.action_model_checkpoint_path)

    if "action_bins" in action_checkpoint_config:
        config.action_bins = action_checkpoint_config["action_bins"]
    if "action_d_model" in action_checkpoint_config:
        config.action_d_model = action_checkpoint_config["action_d_model"]
    if "action_num_transformer_layers" in action_checkpoint_config:
        config.action_num_transformer_layers = action_checkpoint_config[
            "action_num_transformer_layers"
        ]
    if "action_num_heads" in action_checkpoint_config:
        config.action_num_heads = action_checkpoint_config["action_num_heads"]
    if "action_latent_dim" in action_checkpoint_config:
        config.action_latent_dim = action_checkpoint_config["action_latent_dim"]

    if "dynamics_d_model" in dynamics_checkpoint_config:
        config.dynamics_d_model = dynamics_checkpoint_config["dynamics_d_model"]
    if "dynamics_num_transformer_layers" in dynamics_checkpoint_config:
        config.dynamics_num_transformer_layers = dynamics_checkpoint_config[
            "dynamics_num_transformer_layers"
        ]
    if "dynamics_num_heads" in dynamics_checkpoint_config:
        config.dynamics_num_heads = dynamics_checkpoint_config["dynamics_num_heads"]
    if "num_images_in_video" in dynamics_checkpoint_config:
        config.num_images_in_video = dynamics_checkpoint_config["num_images_in_video"]
    if "image_size" in dynamics_checkpoint_config:
        config.image_size = dynamics_checkpoint_config["image_size"]
    if "patch_size" in dynamics_checkpoint_config:
        config.patch_size = dynamics_checkpoint_config["patch_size"]
    if "dataset_type" in dynamics_checkpoint_config:
        config.dataset_type = dynamics_checkpoint_config["dataset_type"]
    if "atari_pong_data_dir" in dynamics_checkpoint_config:
        config.atari_pong_data_dir = dynamics_checkpoint_config["atari_pong_data_dir"]
    if "tokenizer_checkpoint_path" in dynamics_checkpoint_config:
        config.tokenizer_checkpoint_path = dynamics_checkpoint_config[
            "tokenizer_checkpoint_path"
        ]

    return config


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
) -> tuple[str, str, str]:
    """Save all frames as a single horizontal concatenation.
    
    Args:
        frames: (T, C, H, W) or (1, T, C, H, W) tensor
        output_dir: Directory to save to
        prefix: Filename prefix
        step: Step number for filename
    
    Returns:
        Tuple of (archive_path, session_live_path, live_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if frames.dim() == 5:
        frames = frames[0]  # (T, C, H, W)
    
    # Normalize each frame independently
    normalized = torch.stack([_normalize_frame(f) for f in frames], dim=0)
    
    archive_path = output_dir / f"{prefix}_step_{step:03d}.png"
    session_live_path = output_dir / f"{prefix}_live.png"
    live_path = output_dir / "live.png"
    vutils.save_image(normalized.cpu(), archive_path, nrow=normalized.size(0), padding=2)
    vutils.save_image(
        normalized.cpu(), session_live_path, nrow=normalized.size(0), padding=2
    )
    vutils.save_image(normalized.cpu(), live_path, nrow=normalized.size(0), padding=2)
    return str(archive_path), str(session_live_path), str(live_path)


def _get_actual_next_action(
    model, sampled_video: torch.Tensor, required_context: int
) -> Optional[int]:
    if sampled_video.shape[1] <= required_context:
        return None

    with torch.no_grad():
        encoded_actions = model.action_model.encode(sampled_video)
        action_sequence = model.action_model.get_action_sequence(encoded_actions)

    next_action_idx = required_context - 1
    if action_sequence.shape[1] <= next_action_idx:
        return None
    return int(action_sequence[0, next_action_idx].item())


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
    required_context = config.num_images_in_video - 1

    def _sample_new_video() -> tuple[torch.Tensor, Optional[int]]:
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
        actual_next_action = _get_actual_next_action(model, video, required_context)
        return video[:, :required_context, :, :, :], actual_next_action

    # Current sliding window of frames: (1, T, C, H, W)
    current_video: Optional[torch.Tensor] = None
    step_counter = 0
    session_id = 0
    output_dir = Path(args.output_dir)
    total_video: Optional[torch.Tensor] = None
    actual_next_action: Optional[int] = None

    while True:
        # If no current video, sample one
        if current_video is None:
            try:
                current_video, actual_next_action = _sample_new_video()
                total_video = current_video.clone()
                session_id += 1
                step_counter = 0
                prefix = f"session_{session_id:03d}"
                archive_path, session_live_path, live_path = _save_concatenated_frames(
                    current_video, output_dir, prefix, step_counter
                )
                action_summary = (
                    f" actual next action={actual_next_action}"
                    if actual_next_action is not None
                    else ""
                )
                print(
                    f"[session {session_id}] Initial frames saved to: {archive_path}"
                    f" | session live view: {session_live_path} | live view: {live_path}"
                    f"{action_summary}"
                )
            except Exception as exc:
                logger.error("Failed to sample initial video: %s", exc)
                continue

        prompt = "Action id (int)"
        if actual_next_action is not None:
            prompt += f" [actual next action: {actual_next_action}]"
        prompt += ", 'new', or 'q': "
        cmd = input(prompt).strip()
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if cmd.lower() == "new":
            current_video = None
            actual_next_action = None
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
                result = model.predict_next_frame(
                    current_video, action_tensor, max_steps=args.max_steps
                )
            except NotImplementedError:
                logger.warning(
                    "DynamicsModel.predict_next_frame is not implemented."
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
        archive_path, session_live_path, live_path = _save_concatenated_frames(
            total_video, output_dir, prefix, step_counter
        )
        actual_action_summary = (
            str(actual_next_action) if actual_next_action is not None else "n/a"
        )
        print(
            f"[session {session_id}, step {step_counter}] chosen_action={action_id} "
            f"actual_next_action={actual_action_summary} saved to: {archive_path} "
            f"| session live view: {session_live_path} | live view: {live_path}"
        )
        actual_next_action = None


def main():
    args = tyro.cli(InteractiveInferenceArgs)
    device = torch.device(args.device)
    config = _build_config(args)
    model = _load_dynamics_model(config, device)
    test_dataloader = _build_test_dataloader(config)
    _interactive_loop(model, test_dataloader, config, args, device)


if __name__ == "__main__":
    main()

