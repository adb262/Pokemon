import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

action_weight = 1


def compute_target_residuals(video: torch.Tensor) -> torch.Tensor:
    """Return frame-to-frame residual targets with shape [B, T-1, C, H, W]."""
    return video[:, 1:, :, :, :] - video[:, :-1, :, :, :]


def compute_changed_pixel_mask(
    frame_delta: torch.Tensor, threshold: float = 1e-3
) -> torch.Tensor:
    """Mark pixels whose absolute change exceeds the threshold."""
    return (torch.abs(frame_delta) > threshold).float()


def build_weight_mask(
    changed_pixels: torch.Tensor,
    *,
    changed_weight: float,
    unchanged_weight: float,
) -> torch.Tensor:
    """Create a dense weight mask from a binary change map."""
    return torch.where(
        changed_pixels > 0,
        torch.full_like(changed_pixels, changed_weight),
        torch.full_like(changed_pixels, unchanged_weight),
    )


def reconstruction_loss(video: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss that gives higher weight to pixels that changed between frames.
    """
    return F.mse_loss(video, decoded, reduction="mean").mean()


def next_frame_reconstruction_residual_loss(
    video: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruction loss for residual prediction that gives higher weight to pixels that changed between frames.

    Args:
        video: Input video tensor [B, num_images_in_video, C, H, W]
        decoded: Predicted residuals [B, num_images_in_video - 1, C, H, W]

    Returns:
        Weighted reconstruction loss (scalar)
    """
    # Calculate ground truth residuals between consecutive frames
    # [B, num_images_in_video, C, H, W] -> [B, num_images_in_video - 1, C, H, W]
    target_residuals = compute_target_residuals(video)

    # Calculate element-wise MSE loss between predicted and target residuals
    logger.debug(f"decoded shape: {decoded.shape}")
    logger.debug(f"target_residuals shape: {target_residuals.shape}")
    mse_loss = F.mse_loss(
        decoded, target_residuals, reduction="none"
    )  # [B, num_images_in_video - 1, C, H, W]

    # Detect pixels that changed between frames (with small threshold for numerical stability)
    changed_pixels = compute_changed_pixel_mask(
        target_residuals,
        0.005
    )  # [B, num_images_in_video - 1, C, H, W]

    # Create weight mask: higher weight for changed pixels, normal weight for unchanged
    weight_mask = build_weight_mask(
        changed_pixels,
        changed_weight=action_weight,
        unchanged_weight=0,
    )  # [B, num_images_in_video - 1, C, H, W]

    # Apply weights and reduce to scalar
    weighted_loss = (mse_loss * weight_mask).mean()

    return weighted_loss


def next_frame_reconstruction_loss(
    video: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruction loss that gives higher weight to pixels that changed between frames.

    Args:
        video: Video [B, num_images_in_video, C, H, W]
        decoded: Reconstructed frame [B, num_images_in_video - 1, C, H, W]

    Returns:
        Weighted reconstruction loss (scalar)
    """
    # Calculate element-wise MSE loss (no reduction)
    target_frames = video[:, 1:, :, :, :]
    previous_frames = video[:, :-1, :, :, :]

    # Detect pixels that changed between frames (with small threshold for numerical stability)
    changed_pixels = compute_changed_pixel_mask(
        target_frames - previous_frames,
        0.005
    )  # [B, Num_images_in_video - 1, C, H, W]

    # Create weight mask: higher weight for changed pixels, normal weight for unchanged
    weight_mask = build_weight_mask(
        changed_pixels,
        changed_weight=action_weight,
        unchanged_weight=0.0,
    )  # [B, Num_images_in_video - 1, C, H, W]

    # Apply weights and reduce to scalar
    mse_loss = F.mse_loss(
        decoded, target_frames, reduction="none"
    )  # [B, Num_images_in_video - 1, C, H, W]

    return (mse_loss * weight_mask).mean()


def next_frame_reconstruction_loss_l1(
    video: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruction loss that gives higher weight to pixels that changed between frames.

    Args:
        video: Video [B, num_images_in_video, C, H, W]
        decoded: Reconstructed frame [B, num_images_in_video - 1, C, H, W]

    Returns:
        Weighted reconstruction loss (scalar)
    """
    return F.l1_loss(decoded, video, reduction="mean")


def changed_patch_weighted_token_cross_entropy_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    previous_frame_tokens: torch.Tensor,
    current_frame_tokens: torch.Tensor,
    changed_patch_loss_weight: float = 30.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    per_patch_token_loss = F.cross_entropy(
        predicted_tokens.transpose(1, 2),
        target_tokens,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid_targets = target_tokens != ignore_index
    changed_patches = previous_frame_tokens != current_frame_tokens
    patch_weights = torch.ones_like(per_patch_token_loss)
    patch_weights = patch_weights.masked_fill(
        changed_patches & valid_targets, changed_patch_loss_weight
    )
    normalized_weights = patch_weights.masked_fill(~valid_targets, 0)
    return (per_patch_token_loss * patch_weights).sum() / normalized_weights.sum().clamp_min(
        1.0
    )


def clipped_cross_entropy_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    ignore_index: int = -100,
    max_confidence: float = 0.97,
) -> torch.Tensor:
    per_position_loss = F.cross_entropy(
        predicted_tokens.transpose(1, 2),
        target_tokens,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Drop positions where the model is already very confident on any vocab entry.
    max_probs = predicted_tokens.softmax(dim=-1).max(dim=-1).values
    overconfident = max_probs > max_confidence
    valid = target_tokens != ignore_index
    keep = valid & ~overconfident

    return (per_position_loss * keep.float()).sum() / keep.float().sum().clamp_min(1.0)


def clippeßd_l2_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    ignore_index: int = -100,
    min_l2_distance_pixels: float = 10,
) -> torch.Tensor:
    per_position_loss = F.mse_loss(
        predicted_tokens, target_tokens, reduction="none"
    )

    # Drop predictions that are already close to the target in 0-255 pixel space
    # (inputs are assumed to be normalized to [0, 1]). Only train on the pixels
    # we're still meaningfully wrong about.
    pixel_distance = (predicted_tokens - target_tokens).abs() * 255.0
    far_off = pixel_distance > min_l2_distance_pixels
    valid = target_tokens != ignore_index
    keep = valid & far_off

    return (per_position_loss * keep.float()).sum() / keep.float().sum().clamp_min(1.0)