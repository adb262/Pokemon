# Finite Scalar Quantization from https://arxiv.org/pdf/2309.15505
# quantizes each dimension independently by bounding to 0, num_bins then rounding to nearest integer
# prevents token collapse and no auxiliary losses necessary
import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim=5, num_bins=4):
        super().__init__()
        self.num_bins = num_bins  # D
        self.levels_np = torch.tensor(latent_dim * [num_bins])
        self.codebook_size = num_bins**latent_dim  # L^D
        # fsq basis [L^0, L^1, ..., L^(L-1)] for converting between indices and latents
        self.register_buffer(
            "basis", (num_bins ** torch.arange(latent_dim, dtype=torch.long))
        )

    def scale_and_shift(self, z):
        # scale and shift z from [-1, 1] to [0, num_bins - 1]
        return 0.5 * (z + 1) * (self.num_bins - 1)

    def unscale_and_unshift(self, z):
        # unscale and unshift z from [0, num_bins - 1] to [-1, 1]
        return 2 * z / (self.num_bins - 1) - 1

    def forward(self, z):
        # z: [B, T, P, L]
        # apply 0.5 * (tanh(z) + 1) to go from z range to [0, num_bins - 1]
        tanh_z = torch.tanh(z)
        bounded_z = self.scale_and_shift(tanh_z)

        # round to nearest integer
        rounded_z = torch.round(bounded_z)

        # stopgrad for straight-through gradient bypassing quantization
        quantized_z = bounded_z + (rounded_z - bounded_z).detach()

        # normalize back to [-1, 1]
        quantized_z = self.unscale_and_unshift(quantized_z)

        return quantized_z

    def get_codebook_usage(self, quantized_z):
        unique_bins = torch.unique(quantized_z).shape[0]
        return unique_bins / self.num_bins

    def get_indices_from_latents(self, latents, dim=-1):
        # to get fsq indices, for each dimension, we get the index and add it (so multiply by L then sum)
        # for each dimension of each latent, get sum of (value * L^current_dim) along latent dim which is the index of that latent in the codebook
        # codebook size = L^latent_dim
        # latents: [*, L]

        # go from [-1, 1] to [0, num_bins - 1] in each dimension
        digits = torch.round(self.scale_and_shift(latents)).clamp(0, self.num_bins - 1)

        # get indices for each latent by summing (value * L^current_dim_idx) along latent dim
        indices = torch.sum(
            digits * self.basis.to(latents.device), dim=dim
        ).long()  # [*]
        return indices

    def get_latents_from_indices(self, indices, dim=-1):
        # indices: [*]
        # recover each entry of latent in range [0, num_bins - 1] by repeatedly dividing by L^current_dim and taking mod
        digits = (indices.unsqueeze(-1) // self.basis) % self.num_bins  # [*, L]

        # go from [0, num_bins - 1] to [-1, 1] in each dimension
        latents = self.unscale_and_unshift(digits)  # [*, L]
        return latents
