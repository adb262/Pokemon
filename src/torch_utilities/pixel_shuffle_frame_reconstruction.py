import logging

import torch
import torch.nn.functional as F
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

    def zero_init_output(self) -> None:
        """Initialize the output projection to emit exact zeros."""
        nn.init.constant_(self.to_pixels.weight, 0)
        if self.to_pixels.bias is not None:
            nn.init.constant_(self.to_pixels.bias, 0)

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


class UpsampleConvFrameHead(nn.Module):
    """Decode patch-grid tokens with resize-conv blocks instead of subpixel shuffle."""

    def __init__(self, embed_dim: int, hidden_dim: int, patch_size: int, channels: int, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W
        self.Hp, self.Wp = H // patch_size, W // patch_size

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.grid_net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.to_pixels = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)

    def zero_init_output(self) -> None:
        """Initialize the final RGB projection to emit exact zeros."""
        nn.init.constant_(self.to_pixels.weight, 0)
        if self.to_pixels.bias is not None:
            nn.init.constant_(self.to_pixels.bias, 0)

    def forward(self, tokens):  # [B, T, P, E]
        B, T, P, E = tokens.shape
        logger.debug(f"tokens shape: {tokens.shape}")
        x = self.pre_norm(tokens)
        x = rearrange(
            x, "b t (hp wp) e -> (b t) e hp wp", hp=self.Hp, wp=self.Wp
        )
        logger.debug(f"x shape after rearrange: {x.shape}")
        x = self.grid_net(x)
        x = F.interpolate(x, size=(self.H, self.W), mode="nearest")
        x = self.refine(x)
        x = self.to_pixels(x)
        # Squash to [0, 1] so reconstructions live in the valid image range.
        # With zero-bias init, sigmoid(0) = 0.5 -> mid-gray instead of black on the first eval.
        x = torch.sigmoid(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)
        logger.debug(f"x shape after final rearrange: {x.shape}")
        return x
