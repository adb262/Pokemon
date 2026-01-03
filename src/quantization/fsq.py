# Finite Scalar Quantization from https://arxiv.org/pdf/2309.15505
# quantizes each dimension independently by bounding to 0, num_bins then rounding to nearest integer
# prevents token collapse and no auxiliary losses necessary
import logging

import numpy as np
import torch
import torch.nn as nn

from quantization.base import BaseQuantizer

logger = logging.getLogger(__name__)


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(BaseQuantizer):
    def __init__(self, levels: list[int], embedding_dim: int, device: torch.device):
        super().__init__()
        # Keep a simple Python list for length/type checks
        self._levels: list[int] = list(levels)
        # Register levels and derived tensors as buffers so they are saved/restored
        # with the checkpoint and move correctly across devices.
        levels_tensor = torch.tensor(self._levels, dtype=torch.long)
        self.register_buffer("_levels_np", levels_tensor, persistent=True)

        basis = torch.tensor(
            np.concatenate(([1], np.cumprod(levels_tensor[:-1].cpu().numpy()))).astype(
                np.uint32
            ),
            dtype=torch.long,
        )
        self.register_buffer("_basis", basis, persistent=True)

        self._codebook_size: int = int(
            torch.prod(self._levels_np).item()  # type: ignore[arg-type]
        )
        self.device = device

        implicit_codebook = self.indexes_to_codes(
            torch.arange(self._codebook_size, dtype=torch.long)
        )
        self.register_buffer("_implicit_codebook", implicit_codebook, persistent=True)

        # levels_tensor is 1D LongTensor, so len(levels) is a plain int
        self.project_in = nn.Linear(embedding_dim, len(self._levels))

        nn.init.xavier_uniform_(self.project_in.weight)
        if self.project_in.bias is not None:
            nn.init.zeros_(self.project_in.bias)

    def bound(self, z):
        """Bound ‘z‘, an array of shape (..., d)."""
        eps = 1e-3
        levels = self._levels_np.to(z.device)
        half_l = (levels - 1) * (1 - eps) / 2  # type: ignore[operator]
        offset = torch.where(levels % 2 == 1, 0.0, 0.5)  # type: ignore[operator]
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        logger.debug(
            f"Devices: {z.device}, {self.project_in.weight.device}, {self._levels_np.device}"
        )
        logger.debug(f"z shape before project_in: {z.shape}")
        z = self.project_in(z)
        logger.debug(f"z shape after project_in: {z.shape}")
        quantized = round_ste(self.bound(z))
        half_width = self._levels_np.to(z.device) // 2  # type: ignore[operator]
        # Renormalize to [-1, 1]. return quantized / half_width
        logger.debug(f"Quantized: {quantized.shape}, Half width: {half_width.shape}")
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        levels = self._levels_np
        half_width = levels // 2  # type: ignore[operator]
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        levels = self._levels_np
        half_width = levels // 2  # type: ignore[operator]
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        """Converts a ‘code‘ to an index in the codebook."""
        assert zhat.shape[-1] == len(self._levels)
        zhat = self._scale_and_shift(zhat)  # type: ignore[arg-type]
        return (zhat * self._basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor):
        """Inverse of ‘indexes_to_codes‘."""
        indices = indices[..., torch.newaxis]
        long_indices = indices.to(torch.long)
        long_basis = self._basis.to(torch.long)
        long_levels = self._levels_np.to(torch.long)
        # floor_divide and fmod both operate elementwise on tensors
        div = torch.floor_divide(long_indices, long_basis)  # type: ignore[arg-type]
        codes_non_centered = torch.fmod(div, long_levels)  # type: ignore[arg-type]
        return self._scale_and_shift_inverse(codes_non_centered)

    def forward(self, z):
        # z: [B, T, P, L]
        return self.quantize(z)

    def replace_unused_codebooks(self, num_batches: int):
        pass
