import logging

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        *,
        num_images_in_video: int,
        channels: int,
        patch_height: int,
        patch_width: int,
        d_model: int,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels
        patch_dim = channels * patch_height * patch_width

        self._embed_image_patches = nn.Sequential(
            Rearrange(
                "b n c (h p1) (w p2) -> b n (h w) (p1 p2 c)",
                n=num_images_in_video,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, c, h, w)
        logger.debug(
            f"x shape before embed_image_patches in patch embedding: {x.shape}"
        )
        x = self._embed_image_patches(x)
        logger.debug(f"x shape after embed_image_patches: {x.shape}")
        return x


class PatchEmbeddingConv(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int,
        d_model: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_frames, c, h, w)
        b, t, c, h, w = x.shape

        # 1. Merge Batch and Time so Conv2d can process all frames at once
        x = x.view(b * t, c, h, w)

        # 2. Apply Strided Convolution (The "Patch Embedding")
        # Output: (b*t, d_model, h/patch, w/patch)
        x = self.proj(x)

        # 3. Flatten spatial dimensions to get tokens
        # Output: (b*t, d_model, num_patches)
        x = x.flatten(2)

        # 4. Swap dimensions for Transformer (usually expects: Batch, Time, Sequence, Dim)
        # Output: (b*t, num_patches, d_model)
        x = x.transpose(1, 2)

        # 5. Restore Batch and Time dimensions
        # Output: (b, t, num_patches, d_model)
        x = x.view(b, t, -1, x.shape[-1])

        return x
