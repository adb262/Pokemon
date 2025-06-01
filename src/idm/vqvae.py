from typing import Literal
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from codebook import NaiveCodebook
from lapa.positional_bias import Transformer


class VQVAE(nn.Module):
    def __init__(
        self,
        channels: int,
        image_size: tuple[int, int],
        num_patches: int,
        patch_size: tuple[int, int],
        patch_embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_embeddings: int,
        num_heads: int,
        num_transformer_layers: int
    ):
        super(VQVAE, self).__init__()
        self.image_height, self.image_width = image_size
        self.patch_height, self.patch_width = patch_size
        patch_dim = channels * self.patch_height * self.patch_width
        self._embed_image_patches = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',  p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_embed_dim),
            nn.LayerNorm(patch_embed_dim),)
        self.patch_embed_dim = patch_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.project_to_pixels = nn.Sequential(
            nn.LayerNorm(patch_embed_dim),
            nn.Linear(patch_embed_dim, patch_dim),
        )

        # Spatial Transformer for per-frame understanding
        # Will take in (batch_size * 2, num_patches, patch_embed_dim)
        # self.spatial_transformer = nn.Transformer(
        #     d_model=patch_embed_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=6,
        #     num_decoder_layers=6,
        #     dim_feedforward=1024,
        #     activation=F.gelu,
        # )

        # # Temporal Transformer for multi-frame understanding
        # # Will take in (batch_size, num_patches * 2, patch_embed_dim)
        # self.temporal_transformer = nn.Transformer(
        #     d_model=patch_embed_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=6,
        #     num_decoder_layers=6,
        #     dim_feedforward=1024,
        #     activation=F.gelu,
        # )

        self.spatial_transformer = Transformer(
            dim=patch_embed_dim,
            depth=num_transformer_layers,
            dim_head=64,
            heads=num_heads,
            ff_mult=2,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        self.temporal_transformer = Transformer(
            dim=patch_embed_dim,
            depth=num_transformer_layers,
            dim_head=64,
            heads=num_heads,
            ff_mult=2,
        )

        # VQ-VAE codebook
        # self.codebook = NSVQ(
        #     dim=latent_dim,
        #     num_embeddings=num_embeddings,
        #     embedding_dim=embedding_dim,
        #     device=device,
        #     discarding_threshold=0.1,
        #     initialization='normal',
        #     code_seq_len=1,
        #     patch_size=self.patch_height * self.patch_width,
        #     image_size=360,
        # )
        self.codebook = NaiveCodebook(
            num_embeddings,
            patch_embed_dim * num_patches,
            latent_dim
        )

        # Create a sparse projection from patch_embed_dim * num_patches to num_embeddings

        # Decoder to cast from image_0 + action latent space to image_1
        # self.decoder = nn.Transformer(
        #     d_model=patch_embed_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=6,
        #     num_decoder_layers=6,
        #     dim_feedforward=1024,
        # )

        self.decoder = Transformer(
            dim=patch_embed_dim,
            depth=num_transformer_layers,
            dim_head=64,
            heads=num_heads,
            ff_mult=2,
        )

    def encode(self, image_1_batch: torch.Tensor, image_2_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Images are of shape (batch_size, channels, height, width)
        # We want to embed the images into a sequence of patches
        assert image_1_batch.shape == image_2_batch.shape and len(image_1_batch.shape) == 4

        # Embed the images into a sequence of patches
        image_1_patches: torch.Tensor = self._embed_image_patches(image_1_batch)
        image_2_patches: torch.Tensor = self._embed_image_patches(image_2_batch)

        # Encode the patches into a sequence of latent vectors (batch_size, num_patches, patch_embed_dim)
        # Could concat and split up on batch but simpler this way
        encoded_first = self.spatial_transformer(image_1_patches)
        encoded_last = self.spatial_transformer(image_2_patches)

        # Concatenate the two sequences of latent vectors (batch_size, num_patches * 2, patch_embed_dim)
        encoded = torch.cat([encoded_first, encoded_last], dim=1)

        # Pass through the temporal transformer (batch_size, num_patches * 2, patch_embed_dim)
        encoded: torch.Tensor = self.temporal_transformer(encoded)

        encoded_first, encoded_last = encoded.split(encoded.shape[1] // 2, dim=1)
        return encoded_first, encoded_last

    def quantize(self, encoded_first: torch.Tensor, encoded_last: torch.Tensor,
                 mode: Literal["train", "inference"] = "train") -> tuple[torch.Tensor, torch.Tensor]:
        # encoded_first = encoded_first.reshape(encoded_first.shape[0], -1)
        # encoded_last = encoded_last.reshape(encoded_last.shape[0], -1)
        if mode == "train":
            return self.codebook(encoded_first, encoded_last)
        else:
            return self.codebook.inference(encoded_first, encoded_last)

    def decode(self, encoded_first: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        out = self.decoder(encoded_first, context=quantized)
        out = self.project_to_pixels(out)
        return self.reshape_patches_to_images(out)

    def forward(self, image_1_batch: torch.Tensor, image_2_batch: torch.Tensor) -> torch.Tensor:
        encoded_first, encoded_last = self.encode(image_1_batch, image_2_batch)

        # Quantize the latent vectors (batch_size, num_patches, latent_dim)
        quantized, _ = self.quantize(encoded_first, encoded_last)

        # Decode the quantized latent vectors into a sequence of patches (batch_size, num_patches, patch_dim)
        return self.decode(encoded_first.detach(), quantized)

    @torch.inference_mode()
    def inference_step(self, image_1_batch: torch.Tensor, image_2_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_first, encoded_last = self.encode(image_1_batch, image_2_batch)
        quantized, indices = self.quantize(encoded_first, encoded_last, mode="inference")
        return self.decode(encoded_first.detach(), quantized), indices

    def reshape_patches_to_images(self, patches: torch.Tensor) -> torch.Tensor:
        # Calculate number of patches per dimension
        h_patches = self.image_height // self.patch_height
        w_patches = self.image_width // self.patch_width

        return Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_patches, w=w_patches,
                         p1=self.patch_height, p2=self.patch_width, c=3)(patches)
