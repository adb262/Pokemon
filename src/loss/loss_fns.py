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
    Naive residual reconstruction loss over all pixels.

    Args:
        video: Input video tensor [B, num_images_in_video, C, H, W]
        decoded: Predicted residuals [B, num_images_in_video - 1, C, H, W]

    Returns:
        Reconstruction loss (scalar)
    """
    target_residuals = compute_target_residuals(video)
    logger.debug(f"decoded shape: {decoded.shape}")
    logger.debug(f"target_residuals shape: {target_residuals.shape}")
    return F.mse_loss(decoded, target_residuals, reduction="mean")


def next_frame_reconstruction_loss(
    video: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Naive next-frame reconstruction loss over all pixels.

    Args:
        video: Video [B, num_images_in_video, C, H, W]
        decoded: Reconstructed frame [B, num_images_in_video - 1, C, H, W]

    Returns:
        Reconstruction loss (scalar)
    """
    target_frames = video[:, 1:, :, :, :]
    return F.mse_loss(decoded, target_frames, reduction="mean")


def clipped_l2_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    min_l2_distance_pixels: float = 10.0,
) -> torch.Tensor:
    """Train only on pixels whose 0-255 error is larger than the threshold."""
    per_pixel_loss = F.mse_loss(predicted, target, reduction="none")
    pixel_distance = (predicted - target).abs() * 255.0
    keep = pixel_distance > min_l2_distance_pixels
    keep_float = keep.float()
    return (per_pixel_loss * keep_float).sum() / keep_float.sum().clamp_min(1.0)


def clipped_next_frame_reconstruction_loss(
    video: torch.Tensor,
    decoded: torch.Tensor,
    min_l2_distance_pixels: float = 10.0,
) -> torch.Tensor:
    target_frames = video[:, 1:, :, :, :]
    return clipped_l2_reconstruction_loss(
        decoded,
        target_frames,
        min_l2_distance_pixels=min_l2_distance_pixels,
    )


def clipped_next_frame_reconstruction_residual_loss(
    video: torch.Tensor,
    decoded: torch.Tensor,
    min_l2_distance_pixels: float = 10.0,
) -> torch.Tensor:
    target_residuals = compute_target_residuals(video)
    return clipped_l2_reconstruction_loss(
        decoded,
        target_residuals,
        min_l2_distance_pixels=min_l2_distance_pixels,
    )


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


def clipped_l2_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    min_l2_distance_pixels: float = 10,
) -> torch.Tensor:
    return clipped_l2_reconstruction_loss(
        predicted,
        target,
        min_l2_distance_pixels=min_l2_distance_pixels,
    )