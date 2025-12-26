# Finite Scalar Quantization from https://arxiv.org/pdf/2309.15505
# quantizes each dimension independently by bounding to 0, num_bins then rounding to nearest integer
# prevents token collapse and no auxiliary losses necessary
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: list[int], embedding_dim: int, device: torch.device):
        super().__init__()
        self._levels = levels
        self._levels_np = torch.asarray(levels, dtype=torch.long)
        self._basis = torch.tensor(
            np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)
        ).to(torch.long)
        self._levels_np = self._levels_np
        self._codebook_size = torch.prod(self._levels_np).item()
        self.device = device
        logger.info(f"Codebook size: {self._codebook_size}. Levels: {self._levels_np}")
        self._implicit_codebook = self.indexes_to_codes(
            torch.arange(self._codebook_size, dtype=torch.long)
        )
        self.project_in = nn.Linear(embedding_dim, self._levels_np.shape[-1])

    def bound(self, z):
        """Bound ‘z‘, an array of shape (..., d)."""
        eps = 1e-3
        levels = self._levels_np.to(z.device)
        half_l = (levels - 1) * (1 - eps) / 2
        offset = torch.where(levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        logger.info(
            f"Devices: {z.device}, {self.project_in.weight.device}, {self._levels_np.device}"
        )
        z = self.project_in(z)
        quantized = round_ste(self.bound(z))
        half_width = self._levels_np.to(z.device) // 2
        # Renormalize to [-1, 1]. return quantized / half_width
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        """Converts a ‘code‘ to an index in the codebook."""
        assert zhat.shape[-1] == len(self._levels)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor):
        """Inverse of ‘indexes_to_codes‘."""
        indices = indices[..., torch.newaxis]
        long_indices = indices.to(torch.long)
        long_basis = self._basis.to(torch.long)
        long_levels = self._levels_np.to(torch.long)
        codes_non_centered = torch.fmod(
            torch.floor_divide(long_indices, long_basis), long_levels
        )
        return self._scale_and_shift_inverse(codes_non_centered)

    def forward(self, z):
        # z: [B, T, P, L]
        return self.quantize(z)
