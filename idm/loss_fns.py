import torch
import torch.nn.functional as F

action_weight = 4


def reconstruction_loss(image_1: torch.Tensor, image_2: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
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
    mse_loss = F.mse_loss(decoded, image_2, reduction='none')  # [B, C, H, W]

    # Detect pixels that changed between frames (with small threshold for numerical stability)
    threshold = 1e-3
    changed_pixels = (torch.abs(image_1 - image_2) > threshold).float()  # [B, C, H, W]

    # Create weight mask: higher weight for changed pixels, normal weight for unchanged
    weight_mask = torch.where(changed_pixels > 0, action_weight, 1.0)  # [B, C, H, W]

    # Apply weights and reduce to scalar
    weighted_loss = (mse_loss * weight_mask).mean()

    return weighted_loss
