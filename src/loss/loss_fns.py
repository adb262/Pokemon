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
    l2_clip_c: float = 10.0,
) -> torch.Tensor:
    """L2 loss clipped below C, with C expressed in 0-255 pixel units."""
    per_pixel_loss = F.mse_loss(predicted, target, reduction="none")
    clip_c = per_pixel_loss.new_tensor(l2_clip_c / (255.0**2))
    return torch.maximum(per_pixel_loss, clip_c).mean()


def clipped_next_frame_reconstruction_loss(
    video: torch.Tensor,
    decoded: torch.Tensor,
    l2_clip_c: float = 10.0,
) -> torch.Tensor:
    target_frames = video[:, 1:, :, :, :]
    return clipped_l2_reconstruction_loss(
        decoded,
        target_frames,
        l2_clip_c=l2_clip_c,
    )


def clipped_next_frame_reconstruction_residual_loss(
    video: torch.Tensor,
    decoded: torch.Tensor,
    l2_clip_c: float = 10.0,
) -> torch.Tensor:
    target_residuals = compute_target_residuals(video)
    return clipped_l2_reconstruction_loss(
        decoded,
        target_residuals,
        l2_clip_c=l2_clip_c,
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
    ce_clip_c: float = 0.03,
) -> torch.Tensor:
    per_position_loss = F.cross_entropy(
        predicted_tokens.transpose(1, 2),
        target_tokens,
        ignore_index=ignore_index,
        reduction="none",
    )

    valid = target_tokens != ignore_index
    clipped_loss = torch.maximum(
        per_position_loss,
        per_position_loss.new_tensor(ce_clip_c),
    )

    valid_float = valid.float()
    return (clipped_loss * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def clipped_l2_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    l2_clip_c: float = 10.0,
) -> torch.Tensor:
    return clipped_l2_reconstruction_loss(
        predicted,
        target,
        l2_clip_c=l2_clip_c,
    )