import torch
import torch.nn.functional as F

action_weight = 5


def reconstruction_residual_loss(video: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss that gives higher weight to pixels that changed between frames.

    Args:
        image_1: Previous frame [B, C, H, W]
        image_2: Current frame (target) [B, C, H, W] 
        decoded: Reconstructed frame [B, C, H, W]

    Returns:
        Weighted reconstruction loss (scalar)
    """
    # Calculate element-wise MSE loss (no reduction)
    residual = video[:, -1, :, :, :] - video[:, -2, :, :, :]
    mse_loss = F.mse_loss(decoded, residual, reduction='none')  # [B, C, H, W]

    # Detect pixels that changed between frames (with small threshold for numerical stability)
    threshold = 1e-3
    changed_pixels = (torch.abs(residual) > threshold).float()  # [B, C, H, W]

    # Create weight mask: higher weight for changed pixels, normal weight for unchanged
    weight_mask = torch.where(changed_pixels > 0, action_weight, 1.0)  # [B, C, H, W]

    # Apply weights and reduce to scalar
    weighted_loss = (mse_loss * weight_mask).mean()

    return weighted_loss


def reconstruction_loss(video: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss that gives higher weight to pixels that changed between frames.

    Args:
        video: Video [B, num_images_in_video, C, H, W]
        decoded: Reconstructed frame [B, C, H, W]

    Returns:
        Weighted reconstruction loss (scalar)
    """
    # Calculate element-wise MSE loss (no reduction)
    target_frame = video[:, -1, :, :, :]
    video_prefix = video[:, :-1, :, :, :]

    last_frame = video_prefix[:, -1, :, :, :]
    residual = last_frame - target_frame

    # Detect pixels that changed between frames (with small threshold for numerical stability)
    threshold = 1e-3
    changed_pixels = (torch.abs(residual) > threshold).float()  # [B, C, H, W]

    # Create weight mask: higher weight for changed pixels, normal weight for unchanged
    weight_mask = torch.where(changed_pixels > 0, action_weight, 1.0)  # [B, C, H, W]

    # Apply weights and reduce to scalar
    mse_loss = F.mse_loss(decoded, target_frame, reduction='mean', weight=weight_mask)  # [B, C, H, W]

    return mse_loss
