import math

import torch


def _mean_and_var(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and (diagonal) variance over the first dimension.

    Args:
        x: Tensor of shape [N, D] after flattening.
        eps: Small constant to clamp variances for numerical stability.
    """
    # Work in float64 for numerical stability
    x = x.to(torch.float64)
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    var = torch.clamp(var, min=eps)
    return mean, var


def compute_frechet_distance(
    real: torch.Tensor, fake: torch.Tensor, eps: float = 1e-6
) -> float:
    """
    Compute (diagonal) Frechet distance between two sets of samples.

    Args:
        real: Tensor of shape [N, ...]
        fake: Tensor of shape [N, ...], same shape as `real`
        eps: Small value for numerical stability.

    Returns:
        Scalar Frechet distance as a Python float.
    """
    if real.shape != fake.shape:
        raise ValueError(
            f"Real and fake must have the same shape, got {real.shape} and {fake.shape}"
        )

    # Detach and move to CPU to avoid holding computation graph / GPU memory
    real = real.detach().to(torch.float32).cpu()
    fake = fake.detach().to(torch.float32).cpu()

    n = real.shape[0]
    if n == 0:
        return math.inf

    real_flat = real.view(n, -1)
    fake_flat = fake.view(n, -1)

    mu_real, var_real = _mean_and_var(real_flat, eps)
    mu_fake, var_fake = _mean_and_var(fake_flat, eps)

    diff = mu_real - mu_fake
    mean_term = torch.sum(diff * diff)

    # Diagonal covariance version of the Frechet distance
    cov_term = torch.sum(
        var_real
        + var_fake
        - 2.0 * torch.sqrt(torch.clamp(var_real * var_fake, min=eps))
    )

    frechet_distance = mean_term + cov_term
    if not torch.isfinite(frechet_distance):
        return math.inf

    return float(frechet_distance.item())
