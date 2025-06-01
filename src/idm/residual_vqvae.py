from typing import Literal
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from lapa.positional_bias import Transformer
from vector_quantize_pytorch import ResidualLFQ


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

        # ResidualLFQ configuration - each LFQ layer needs dim = log2(codebook_size)

        self.residual_lfq = ResidualLFQ(
            dim=latent_dim,  # Input dimension (will be projected to codebook_dim internally)
            codebook_size=patch_embed_dim,  # 256 codes per quantizer
            num_quantizers=num_embeddings,  # Number of residual quantizers
            quantize_dropout=True,  # Enable dropout for better generalization
            quantize_dropout_cutoff_index=1,  # Keep at least 1 quantizer
            experimental_softplus_entropy_loss=True,
            entropy_loss_weight=0.1,
            commitment_loss_weight=0.25,
        )

        # Project difference to action space
        self.to_action = nn.Sequential(
            nn.LayerNorm(patch_embed_dim),
            nn.Linear(patch_embed_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Project action back to patch space for decoding
        self.from_action = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, patch_embed_dim),
            nn.LayerNorm(patch_embed_dim),
        )

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
                 mode: Literal["train", "inference"] = "train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the difference (action) between frames
        # Shape: (batch_size, num_patches, patch_embed_dim)
        frame_diff = encoded_last - encoded_first

        # Pool across patches to get a global action representation
        # You could also use attention pooling here
        global_action = frame_diff.mean(dim=1)  # (batch_size, patch_embed_dim)

        # Project to action space
        action_features = self.to_action(global_action)  # (batch_size, action_dim)

        # ResidualLFQ expects 3D input: (batch, sequence, dim)
        # We treat each action as a single "token" in the sequence
        action_features = action_features.unsqueeze(1)  # (batch_size, 1, action_dim)

        # Quantize the action
        quantized_action, indices, aux_loss = self.residual_lfq(action_features)

        # Remove sequence dimension
        quantized_action = quantized_action.squeeze(1)  # (batch_size, action_dim)

        # Project back to patch space and broadcast to all patches
        decoded_action = self.from_action(quantized_action)  # (batch_size, patch_embed_dim)
        # (batch_size, num_patches, patch_embed_dim)
        decoded_action = decoded_action.unsqueeze(1).expand(-1, frame_diff.size(1), -1)

        return decoded_action, indices, aux_loss

    def decode(self, encoded_first: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        out = self.decoder(encoded_first, context=quantized)
        out = self.project_to_pixels(out)
        return self.reshape_patches_to_images(out)

    def forward(self, image_1_batch: torch.Tensor, image_2_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_first, encoded_last = self.encode(image_1_batch, image_2_batch)

        # Quantize the latent vectors (batch_size, num_patches, latent_dim)
        quantized, indices, commit_loss = self.quantize(encoded_first, encoded_last)

        # Decode the quantized latent vectors into a sequence of patches (batch_size, num_patches, patch_dim)
        return self.decode(encoded_first.detach(), quantized), commit_loss

    @torch.inference_mode()
    def inference_step(self, image_1_batch: torch.Tensor, image_2_batch: torch.Tensor) -> torch.Tensor:
        encoded_first, encoded_last = self.encode(image_1_batch, image_2_batch)
        quantized, indices, commit_loss = self.quantize(encoded_first, encoded_last, mode="inference")
        return self.decode(encoded_first.detach(), quantized)

    def reshape_patches_to_images(self, patches: torch.Tensor) -> torch.Tensor:
        # Calculate number of patches per dimension
        h_patches = self.image_height // self.patch_height
        w_patches = self.image_width // self.patch_width

        return Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_patches, w=w_patches,
                         p1=self.patch_height, p2=self.patch_width, c=3)(patches)
