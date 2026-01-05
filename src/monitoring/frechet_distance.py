import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision import models, transforms

# ============================================================================
# FID (Frechet Inception Distance) - Image-based metric
# ============================================================================


@lru_cache(maxsize=1)
def _get_inception_model(device: torch.device) -> nn.Module:
    """
    Load pretrained Inception-v3 and modify it for feature extraction.
    Uses lru_cache to avoid reloading the model multiple times.
    """
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

    inception.fc = nn.Identity()  # type: ignore
    inception.eval()
    inception.to(device)
    return inception


def _get_inception_transforms() -> transforms.Compose:
    """Get the preprocessing transforms for Inception-v3."""
    return transforms.Compose(
        [
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _extract_inception_features(
    images: torch.Tensor,
    model: nn.Module,
    transform: transforms.Compose,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Extract Inception-v3 features from a batch of images.

    Args:
        images: Tensor of shape [N, C, H, W] with values in [0, 1] or [-1, 1]
        model: Pretrained Inception model
        transform: Preprocessing transforms
        batch_size: Batch size for feature extraction

    Returns:
        Tensor of shape [N, 2048] containing Inception features
    """
    device = next(model.parameters()).device

    # Normalize to [0, 1] if needed (assuming input might be in [-1, 1])
    if images.min() < 0:
        images = (images + 1) / 2

    # Clamp to valid range
    images = images.clamp(0, 1)

    features_list = []
    n_samples = images.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = images[i : i + batch_size].to(device)
            # Apply transforms (resize and normalize)
            batch = transform(batch)
            # Extract features
            feats = model(batch)
            features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


def _compute_statistics(
    features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.

    Args:
        features: Tensor of shape [N, D]

    Returns:
        Tuple of (mean, covariance) as numpy arrays
    """
    features_np = features.numpy().astype(np.float64)
    mu = np.mean(features_np, axis=0)
    sigma = np.cov(features_np, rowvar=False)
    return mu, sigma


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Frechet distance between two multivariate Gaussians.

    The Frechet distance is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
        eps: Small value for numerical stability

    Returns:
        Frechet distance as a float
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    # Add a small offset to the covariance matrices to improve numerical stability
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    if not np.isfinite(fid):
        return math.inf

    return float(fid)


def _video_to_frames(video: torch.Tensor) -> torch.Tensor:
    """
    Convert video tensor to frames tensor.

    Args:
        video: Tensor of shape [B, T, C, H, W] or [B, T, H, W, C]

    Returns:
        Tensor of shape [B*T, C, H, W]
    """
    if video.dim() != 5:
        raise ValueError(f"Expected 5D video tensor, got shape {video.shape}")

    b, t = video.shape[:2]

    # Check if channels are last (H, W, C) or first (C, H, W)
    # Assume if dim 2 is 3 or 1, it's channels-first
    if video.shape[2] in (1, 3):
        # [B, T, C, H, W] -> [B*T, C, H, W]
        frames = video.view(b * t, *video.shape[2:])
    else:
        # [B, T, H, W, C] -> [B, T, C, H, W] -> [B*T, C, H, W]
        video = video.permute(0, 1, 4, 2, 3)
        frames = video.view(b * t, *video.shape[2:])

    return frames


def compute_frechet_distance(
    real: torch.Tensor,
    fake: torch.Tensor,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> float:
    """
    Compute Frechet Inception Distance (FID) between real and fake samples.

    This uses a pretrained Inception-v3 model to extract features, then
    computes the Frechet distance between the feature distributions.

    Args:
        real: Tensor of shape [N, C, H, W] for images or [N, T, C, H, W] for videos
        fake: Tensor of shape matching real
        batch_size: Batch size for feature extraction
        device: Device to run Inception on (defaults to cuda if available)

    Returns:
        FID score as a float (lower is better)
    """
    if real.shape != fake.shape:
        raise ValueError(
            f"Real and fake must have the same shape, got {real.shape} and {fake.shape}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to float and detach
    real = real.detach().float()
    fake = fake.detach().float()

    # Handle video input by converting to frames
    if real.dim() == 5:
        real = _video_to_frames(real)
        fake = _video_to_frames(fake)
    elif real.dim() != 4:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {real.shape}")

    n_samples = real.shape[0]
    if n_samples < 2:
        return math.inf

    # Load model and transforms
    model = _get_inception_model(device)
    transform = _get_inception_transforms()

    # Extract features
    real_features = _extract_inception_features(real, model, transform, batch_size)
    fake_features = _extract_inception_features(fake, model, transform, batch_size)

    # Compute statistics
    mu_real, sigma_real = _compute_statistics(real_features)
    mu_fake, sigma_fake = _compute_statistics(fake_features)

    # Calculate FID
    fid = _calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return fid


# ============================================================================
# FVD (Frechet Video Distance) - Video-based metric using I3D features
# ============================================================================

# I3D normalization constants (Kinetics-400 dataset statistics)
_I3D_MEAN = [0.43216, 0.394666, 0.37645]
_I3D_STD = [0.22803, 0.22145, 0.216989]

# Expected temporal length for I3D (can process variable lengths, but 16 is standard)
_I3D_TEMPORAL_LENGTH = 16
_I3D_SPATIAL_SIZE = 224


@lru_cache(maxsize=1)
def _get_i3d_model(device: torch.device) -> nn.Module:
    """
    Load pretrained I3D-style model (R3D-18) for video feature extraction.
    Uses lru_cache to avoid reloading the model multiple times.

    We use R3D-18 (3D ResNet) from torchvision which is trained on Kinetics-400,
    similar to the original I3D used in the FVD paper.
    """
    # R3D-18 is a 3D ResNet trained on Kinetics-400
    model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)

    # Replace the final FC layer with identity for feature extraction
    # R3D-18 outputs 512-dimensional features before the classifier
    model.fc = nn.Identity()  # type: ignore
    model.eval()
    model.to(device)
    return model


def _get_i3d_transforms() -> transforms.Compose:
    """Get the preprocessing transforms for I3D/R3D models."""
    return transforms.Compose(
        [
            transforms.Normalize(mean=_I3D_MEAN, std=_I3D_STD),
        ]
    )


def _preprocess_video_for_i3d(
    video: torch.Tensor,
    target_frames: int = _I3D_TEMPORAL_LENGTH,
    target_size: int = _I3D_SPATIAL_SIZE,
) -> torch.Tensor:
    """
    Preprocess video tensor for I3D feature extraction.

    Args:
        video: Tensor of shape [B, T, C, H, W] with values in [0, 1] or [-1, 1]
        target_frames: Number of frames to sample (default 16)
        target_size: Spatial size to resize to (default 224)

    Returns:
        Tensor of shape [B, C, T, H, W] ready for I3D (note: channels before time)
    """
    b, t, c, h, w = video.shape

    # Normalize to [0, 1] if needed
    if video.min() < 0:
        video = (video + 1) / 2
    video = video.clamp(0, 1)

    # Temporal sampling: uniformly sample target_frames from the video
    if t != target_frames:
        # Use linear interpolation along temporal dimension
        # Reshape to [B*C, 1, T, H*W] for temporal interpolation
        video_flat = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        video_flat = video_flat.reshape(b, c, t, h * w)  # [B, C, T, H*W]

        # Interpolate temporally
        video_flat = F.interpolate(
            video_flat,
            size=(target_frames, h * w),
            mode="bilinear",
            align_corners=False,
        )
        video = video_flat.reshape(b, c, target_frames, h, w)
        video = video.permute(0, 2, 1, 3, 4)  # Back to [B, T, C, H, W]
        t = target_frames

    # Spatial resize if needed
    if h != target_size or w != target_size:
        # Reshape to [B*T, C, H, W] for spatial resize
        video_flat = video.reshape(b * t, c, h, w)
        video_flat = F.interpolate(
            video_flat,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        video = video_flat.reshape(b, t, c, target_size, target_size)

    # Convert from [B, T, C, H, W] to [B, C, T, H, W] for 3D CNN
    video = video.permute(0, 2, 1, 3, 4)

    return video


def _extract_i3d_features(
    videos: torch.Tensor,
    model: nn.Module,
    transform: transforms.Compose,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Extract I3D/R3D features from a batch of videos.

    Args:
        videos: Tensor of shape [N, T, C, H, W] with values in [0, 1] or [-1, 1]
        model: Pretrained I3D/R3D model
        transform: Preprocessing transforms (normalization)
        batch_size: Batch size for feature extraction

    Returns:
        Tensor of shape [N, 512] containing video features
    """
    device = next(model.parameters()).device

    # Preprocess videos (resize, temporal sampling, channel reorder)
    videos = _preprocess_video_for_i3d(videos)  # Now [N, C, T, H, W]

    features_list = []
    n_samples = videos.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = videos[i : i + batch_size].to(device)

            # Apply normalization transform to each frame
            # batch is [B, C, T, H, W], need to normalize along C dimension
            b, c, t, h, w = batch.shape
            batch = batch.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            batch = batch.reshape(b * t, c, h, w)  # [B*T, C, H, W]
            batch = transform(batch)  # Apply normalization
            batch = batch.reshape(b, t, c, h, w)  # [B, T, C, H, W]
            batch = batch.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            # Extract features
            feats = model(batch)
            features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


def _ensure_video_format(video: torch.Tensor) -> torch.Tensor:
    """
    Ensure video is in [B, T, C, H, W] format.

    Args:
        video: Tensor of shape [B, T, C, H, W] or [B, T, H, W, C]

    Returns:
        Tensor of shape [B, T, C, H, W]
    """
    if video.dim() != 5:
        raise ValueError(f"Expected 5D video tensor, got shape {video.shape}")

    # Check if channels are last (H, W, C) or first (C, H, W)
    # Assume if dim 2 is 3 or 1, it's channels-first
    if video.shape[2] not in (1, 3):
        # [B, T, H, W, C] -> [B, T, C, H, W]
        video = video.permute(0, 1, 4, 2, 3)

    return video


def compute_fvd(
    real: torch.Tensor,
    fake: torch.Tensor,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> float:
    """
    Compute Frechet Video Distance (FVD) between real and fake video samples.

    This uses a pretrained I3D-style model (R3D-18 trained on Kinetics-400) to
    extract spatio-temporal features, then computes the Frechet distance between
    the feature distributions.

    FVD is the video equivalent of FID and captures both spatial quality and
    temporal coherence of generated videos.

    Args:
        real: Tensor of shape [N, T, C, H, W] or [N, T, H, W, C] for videos
        fake: Tensor of shape matching real
        batch_size: Batch size for feature extraction (lower than FID due to
                    memory requirements of 3D convolutions)
        device: Device to run I3D on (defaults to cuda if available)

    Returns:
        FVD score as a float (lower is better)

    Example:
        >>> real_videos = torch.randn(100, 16, 3, 64, 64)  # 100 videos, 16 frames
        >>> fake_videos = torch.randn(100, 16, 3, 64, 64)
        >>> fvd = compute_fvd(real_videos, fake_videos)
    """
    if real.dim() != 5 or fake.dim() != 5:
        raise ValueError(
            f"FVD requires 5D video tensors [B, T, C, H, W], "
            f"got real: {real.shape}, fake: {fake.shape}"
        )

    if real.shape != fake.shape:
        raise ValueError(
            f"Real and fake must have the same shape, got {real.shape} and {fake.shape}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to float and detach
    real = real.detach().float()
    fake = fake.detach().float()

    # Ensure correct format [B, T, C, H, W]
    real = _ensure_video_format(real)
    fake = _ensure_video_format(fake)

    n_samples = real.shape[0]
    if n_samples < 2:
        return math.inf

    # Load model and transforms
    model = _get_i3d_model(device)
    transform = _get_i3d_transforms()

    # Extract features
    real_features = _extract_i3d_features(real, model, transform, batch_size)
    fake_features = _extract_i3d_features(fake, model, transform, batch_size)

    # Compute statistics (reuse from FID)
    mu_real, sigma_real = _compute_statistics(real_features)
    mu_fake, sigma_fake = _compute_statistics(fake_features)

    # Calculate FVD (same formula as FID)
    fvd = _calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return fvd
