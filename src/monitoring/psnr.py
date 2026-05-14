"""PSNR (Peak Signal-to-Noise Ratio) and Δt PSNR controllability metrics.

The Δt PSNR metric is from the Genie paper and measures controllability:
    Δt_PSNR = PSNR(x_t, x̂_t) - PSNR(x_t, x̂'_t)

Where:
    - x_t: ground-truth frame at time t
    - x̂_t: frame generated from latent actions inferred from ground-truth frames
    - x̂'_t: frame generated from randomly sampled latent actions

A higher Δt PSNR indicates better controllability (inferred actions produce
better generations than random ones).
"""

import logging
import math

import torch

from dynamics_model.model import DynamicsModel

logger = logging.getLogger(__name__)


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two tensors.

    Args:
        original: Ground-truth tensor with values in [0, max_value]
        reconstructed: Reconstructed tensor with values in [0, max_value]
        max_value: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        Average PSNR in dB (higher is better)
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )

    # Normalize to [0, 1] if values are in [-1, 1]
    if original.min() < 0:
        original = (original + 1) / 2
    if reconstructed.min() < 0:
        reconstructed = (reconstructed + 1) / 2

    # Clamp to valid range
    original = original.clamp(0, max_value)
    reconstructed = reconstructed.clamp(0, max_value)

    # Compute MSE per sample
    mse = torch.mean((original - reconstructed) ** 2)

    if mse < 1e-10:
        return float("inf")

    # PSNR = 10 * log10(MAX^2 / MSE)
    psnr = 10 * torch.log10(max_value**2 / mse)

    return float(psnr.item())


def compute_frame_pixel_similarity(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
) -> float:
    """
    Compute mean per-sample cosine similarity between two batches of frames.

    Treats each frame as a flattened pixel vector. Bounded in [-1, 1] where 1.0
    means identical frames (up to scale) and 0.0 means orthogonal.

    Args:
        frame_a: Tensor of shape [B, C, H, W]
        frame_b: Tensor of shape [B, C, H, W]

    Returns:
        Mean cosine similarity across the batch.
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError(
            f"Shape mismatch: frame_a {frame_a.shape} vs frame_b {frame_b.shape}"
        )

    a = frame_a.flatten(1).float()
    b = frame_b.flatten(1).float()
    sim = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return float(sim.mean().item())


def compute_psnr_per_frame(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> list[float]:
    """
    Compute PSNR for each frame in a video batch.

    Args:
        original: Ground-truth video [B, T, C, H, W]
        reconstructed: Reconstructed video [B, T, C, H, W]
        max_value: Maximum possible pixel value

    Returns:
        List of PSNR values, one per frame position
    """
    if original.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, T, C, H, W], got {original.shape}")

    num_frames = original.shape[1]
    psnr_per_frame = []

    for t in range(num_frames):
        psnr = compute_psnr(original[:, t], reconstructed[:, t], max_value)
        psnr_per_frame.append(psnr)

    return psnr_per_frame


@torch.no_grad()
def compute_delta_psnr(
    model: DynamicsModel,
    video_batch: torch.Tensor,
    device: torch.device,
    max_steps: int = 25,
) -> tuple[float, float, float]:
    """
    Compute the Δt PSNR controllability metric from the Genie paper.

    This metric measures how much the video generations differ when conditioned
    on latent actions inferred from ground-truth vs. sampled from a random
    distribution.

    Args:
        model: The dynamics model with a latent-action model attached.
        video_batch: Input video tensor [B, T, C, H, W]
        device: Device to run computation on
        max_steps: Number of MaskGIT decoding steps for each generated frame.

    Returns:
        tuple of (psnr_inferred, psnr_random, delta_psnr)
        - psnr_inferred: PSNR using actions inferred from ground-truth
        - psnr_random: PSNR using randomly sampled actions
        - delta_psnr: psnr_inferred - psnr_random (higher = better controllability)
    """
    video_batch = video_batch.to(device)

    # Use the inferred action for the transition into the held-out final frame.
    quantized_inferred = model.action_model.encode(video_batch)
    inferred_actions = model.action_model.get_action_sequence(quantized_inferred)
    inferred_last_action = inferred_actions[:, -1]

    action_vocab_size = model.action_model.action_vocab_size
    random_last_action = torch.randint(
        0,
        action_vocab_size,
        (video_batch.shape[0],),
        device=device,
        dtype=torch.long,
    )
    if action_vocab_size > 1:
        random_last_action = torch.where(
            random_last_action == inferred_last_action,
            (random_last_action + 1) % action_vocab_size,
            random_last_action,
        )

    seed_video = video_batch[:, :-1]
    reconstructed_inferred = model.rollout(
        seed_video,
        inferred_last_action.unsqueeze(1),
        max_steps=max_steps,
    )
    reconstructed_random = model.rollout(
        seed_video,
        random_last_action.unsqueeze(1),
        max_steps=max_steps,
    )

    real_target = video_batch[:, -1]
    inferred_target = reconstructed_inferred[:, -1]
    random_target = reconstructed_random[:, -1]

    psnr_inferred = compute_psnr(real_target, inferred_target)
    psnr_random = compute_psnr(real_target, random_target)
    delta_psnr = psnr_inferred - psnr_random

    return psnr_inferred, psnr_random, delta_psnr


@torch.no_grad()
def compute_delta_psnr_batched(
    model: DynamicsModel,
    video_batches: list[torch.Tensor],
    device: torch.device,
    max_steps: int = 25,
) -> dict[str, float]:
    """
    Compute Δt PSNR metrics over multiple batches.

    Args:
        model: The dynamics model with a latent-action model attached.
        video_batches: List of video batch tensors
        device: Device to run computation on
        max_steps: Number of MaskGIT decoding steps for each generated frame.

    Returns:
        Dictionary with averaged metrics:
        - psnr_inferred: Average PSNR with inferred actions
        - psnr_random: Average PSNR with random actions
        - delta_psnr: Average Δt PSNR
    """
    if not video_batches:
        return {
            "psnr_inferred": math.inf,
            "psnr_random": math.inf,
            "delta_psnr": 0.0,
        }

    psnr_inferred_list = []
    psnr_random_list = []
    delta_psnr_list = []

    for video_batch in video_batches:
        psnr_inf, psnr_rand, delta = compute_delta_psnr(
            model,
            video_batch,
            device,
            max_steps=max_steps,
        )
        # Skip infinite values
        if math.isfinite(psnr_inf) and math.isfinite(psnr_rand):
            psnr_inferred_list.append(psnr_inf)
            psnr_random_list.append(psnr_rand)
            delta_psnr_list.append(delta)

    if not psnr_inferred_list:
        return {
            "psnr_inferred": math.inf,
            "psnr_random": math.inf,
            "delta_psnr": 0.0,
        }

    return {
        "psnr_inferred": sum(psnr_inferred_list) / len(psnr_inferred_list),
        "psnr_random": sum(psnr_random_list) / len(psnr_random_list),
        "delta_psnr": sum(delta_psnr_list) / len(delta_psnr_list),
    }


