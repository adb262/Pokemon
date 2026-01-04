import torch
import torch.nn as nn


class MaskGIT(nn.Module):
    def __init__(self, mask_ratio: float, d_model: int):
        super(MaskGIT, self).__init__()
        self.mask_ratio = mask_ratio
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, num_patches, d_model)
        batch_size, num_images_in_video, num_patches, d_model = x.shape

        # x is of shape (batch_size, num_images_in_video, num_patches, d_model)
        # We want to mask out some of the patches
        mask = torch.rand(x.shape[0], x.shape[1]) < self.mask_ratio
        x[mask] = self.mask_token
        return x
