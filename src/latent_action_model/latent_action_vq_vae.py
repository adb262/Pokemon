import logging

import torch
import torch.nn as nn

from latent_action_model.patch_embedding import PatchEmbeddingConv
from quantization.fsq import FiniteScalarQuantizer
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer
from video_tokenization.tokenizer import PixelShuffleFrameHead

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ActionCondenserNet(nn.Module):
    def __init__(
        self,
        num_patches: int,
        d_model_input: int,
        d_model_output: int,
        d_model_intermediate_factor: int = 1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.d_model_input = d_model_input
        d_model_intermediate = d_model_input * d_model_intermediate_factor

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model_input,
                out_channels=d_model_intermediate,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=d_model_intermediate,
                out_channels=d_model_intermediate,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(d_model_intermediate, d_model_output)
        self.ln = nn.LayerNorm(d_model_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, P, D_in)
        batch_size, num_frames, num_patches, d_model = x.shape
        x = x.view(batch_size * num_frames, d_model, num_patches)  # (B * T, D_in, P)
        x = self.conv_block(x)  # (B * T, D_inter, 1)
        x = x.squeeze(-1).view(batch_size, num_frames, d_model)  # (B, T, D_inter)
        logger.debug(f"x shape after conv_block: {x.shape}")
        x = self.fc(x)  # (B, T, D_out)
        x = self.ln(x)  # (B, T, D_out)
        return x


class LatentActionVQVAE(nn.Module):
    def __init__(
        self,
        *,
        num_images_in_video: int,
        image_height: int,
        image_width: int,
        channels: int,
        patch_height: int,
        patch_width: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        use_spatial_transformer: bool,
        use_temporal_transformer: bool,
        num_embeddings: int,
        embedding_dim: int,
        bins: list[int] = [8],  # vocab size of 8
    ):
        super(LatentActionVQVAE, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.embed_image_patches = PatchEmbeddingConv(
            channels=channels,
            patch_size=patch_height,
            d_model=d_model,
        )

        self.encoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=use_spatial_transformer,
            use_temporal_transformer=use_temporal_transformer,
        )

        num_patches_h = image_height // patch_height
        num_patches_w = image_width // patch_width
        self.num_patches_per_image = num_patches_h * num_patches_w

        self.action_condenser = ActionCondenserNet(
            num_patches=self.num_patches_per_image,
            d_model_input=d_model,
            d_model_output=d_model,
        )

        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # self.quantizer = NSVQ(
        #     dim=d_model,
        #     num_embeddings=num_embeddings,
        #     embedding_dim=embedding_dim,
        #     patch_size=patch_height,
        #     image_size=image_height,
        #     device=device,
        #     discarding_threshold=0.1,
        #     initialization="normal",
        #     code_seq_len=1,
        # )

        self.decoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.patch_to_pixels = PixelShuffleFrameHead(
            embed_dim=d_model,
            patch_size=patch_height,
            channels=channels,
            H=image_height,
            W=image_width,
        )

        self.quantizer = FiniteScalarQuantizer(
            levels=bins,
            embedding_dim=d_model,
            device=device,
        )

        # This will act as a glorified embedding layer for the action
        self.action_projection = nn.Linear(1, d_model)

        # self.decoder = VideoDecoder(
        #     num_images_in_video=num_images_in_video - 1,
        #     image_height=image_height,
        #     image_width=image_width,
        #     channels=channels,
        #     patch_height=patch_height,
        #     patch_width=patch_width,
        #     d_model=d_model,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     use_spatial_transformer=use_spatial_transformer,
        #     use_temporal_transformer=use_temporal_transformer,
        # )

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"LatentActionVQVAE initialized. Num params: {params}")

        # Initialize everything with Xavier initialization
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

        for p in self.action_projection.parameters():
            nn.init.xavier_uniform_(p)

        for p in self.quantizer.parameters():
            nn.init.xavier_uniform_(p)

        for p in self.decoder.parameters():
            nn.init.xavier_uniform_(p)

        for p in self.patch_to_pixels.parameters():
            nn.init.xavier_uniform_(p)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # We want to encode the video into a sequence of patches
        logger.debug(f"video shape: {video.shape}")
        # patched_video_from_embedder has shape (B, T*P, D) then reshaped to (B, T, P, D) by PatchEmbedding
        patched_video_from_embedder = self.embed_image_patches(video)
        logger.debug(
            f"patched_video_from_embedder shape: {patched_video_from_embedder.shape}"
        )

        # x_encoded_full has shape (batch_size, num_frames, num_patches, d_model)
        x_encoded_full = self.encoder(patched_video_from_embedder)
        logger.debug(f"x_encoded_full shape after encoder: {x_encoded_full.shape}")

        # Pass in second to last frame and last frame
        action_differential = x_encoded_full[:, 1:, :, :] - x_encoded_full[:, :-1, :, :]
        x_encoded_mean = action_differential.mean(dim=2)

        logger.debug(f"x_encoded_mean shape: {x_encoded_mean.shape}")
        quantized = self.quantizer(x_encoded_mean)

        # quantized, indices, commitment_loss = self.quantizer(action_continuous)
        logger.debug(f"quantized shape: {quantized.shape}")
        logger.info(f"quantized values: {quantized}")

        # Project the action to the d_model space
        action_projected = self.action_projection(quantized)
        logger.info(f"action_projected shape: {action_projected.shape}")

        # Trim off the final image from the video for decoder input
        # Patches for frames 0 to T-2
        patch_video_for_decoder = patched_video_from_embedder[
            :, :-1, :, :
        ] + action_projected.unsqueeze(2)

        # Now, we need to decode. The decoder aims to predict frame T-1 using frames 0 to T-2 and the quantized_action.
        decoded = self.decoder(patch_video_for_decoder)

        # Shape is (B, N_images_in_video - 1, num_patches, d_model)
        logger.debug(
            f"decoded shape: {decoded.shape}"
        )  # Decoder outputs a single frame (B, C, H, W)
        decoded = self.patch_to_pixels(decoded)
        logger.debug(f"decoded shape after patch_to_pixels: {decoded.shape}")
        return decoded, quantized
