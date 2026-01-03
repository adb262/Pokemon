import logging

import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PixelShuffleFrameHead(nn.Module):
    # conv2D embeddings to pixels head
    def __init__(self, embed_dim, patch_size=8, channels=3, H=128, W=128):
        super().__init__()
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.to_pixels = nn.Conv2d(embed_dim, channels * (patch_size**2), kernel_size=1)

    def forward(self, tokens):  # [B, T, P, E]
        B, T, P, E = tokens.shape
        logger.debug(f"tokens shape: {tokens.shape}")
        x = rearrange(
            tokens, "b t (hp wp) e -> (b t) e hp wp", hp=self.Hp, wp=self.Wp
        )  # [(B*T), E, Hp, Wp]
        logger.debug(f"x shape after rearrange: {x.shape}")
        x = self.to_pixels(x)  # [(B*T), C*p^2, Hp, Wp]
        logger.debug(f"x shape after to_pixels: {x.shape}")
        x = rearrange(
            x,
            "(b t) (c p1 p2) hp wp -> b t c (hp p1) (wp p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            b=B,
            t=T,
        )  # [B, T, C, H, W]
        logger.debug(f"x shape after final rearrange: {x.shape}")
        return x
