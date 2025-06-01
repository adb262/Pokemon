from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, *, num_images_in_video: int, channels: int, patch_height: int, patch_width: int, d_model: int):
        super(PatchEmbedding, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels
        patch_dim = channels * patch_height * patch_width

        self._embed_image_patches = nn.Sequential(
            Rearrange(
                'b n c (h p1) (w p2) -> b n (h w) (p1 p2 c)', n=num_images_in_video, p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embed_image_patches(x)
