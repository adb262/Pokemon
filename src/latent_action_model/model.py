import logging
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from latent_action_model.patch_embedding import PatchEmbedding, PatchEmbeddingConv
from quantization.base import BaseQuantizer
from quantization.fsq import FiniteScalarQuantizer
from quantization.nsvq import NSVQ
from torch_utilities.crop_center_patches import get_center_patch_indices
from torch_utilities.initialize import init_weights
from torch_utilities.pixel_shuffle_frame_reconstruction import UpsampleConvFrameHead
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LatentActionVQVAE(nn.Module):
    quantizer: BaseQuantizer
    action_vocab_size: int

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
        embedding_dim: int,
        quantizer_type: Literal["nsvq", "fsq"] = "fsq",
        bins: list[int] = [8],  # vocab size of 8
        zero_init_output_head: bool = False,
    ):
        super(LatentActionVQVAE, self).__init__()


        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.embed_image_patches = PatchEmbedding(
            num_images_in_video=num_images_in_video,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
        )

        self.encoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )

        self.embed_image_patches_decoder = PatchEmbeddingConv(
            channels=channels,
            patch_size=patch_height,
            d_model=d_model,
        )

        num_patches_h = image_height // patch_height
        num_patches_w = image_width // patch_width
        self.num_patches_per_image = num_patches_h * num_patches_w

        self.quantizer_type = quantizer_type

        if quantizer_type == "nsvq":
            self.quantizer = NSVQ(
                dim=d_model,
                num_embeddings=len(bins),
                embedding_dim=embedding_dim,
                patch_size=patch_height,
                image_size=image_height,
                device=device,
                discarding_threshold=0.1,
                initialization="normal",
                code_seq_len=1,
            )
        elif quantizer_type == "fsq":
            self.quantizer = FiniteScalarQuantizer(
                levels=bins,
                embedding_dim=embedding_dim,
                device=device,
            )
        else:
            raise ValueError(f"Invalid quantizer type: {quantizer_type}")

        self.action_vocab_size = self.quantizer.codebook_size
        self.d_model = d_model
        self.decoder_transformer = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.patch_to_pixels = UpsampleConvFrameHead(
            embed_dim=d_model,
            hidden_dim=d_model // 2,
            patch_size=patch_height,
            channels=channels,
            H=image_height,
            W=image_width,
        )

        self.pre_action_norm = nn.RMSNorm(d_model)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, len(bins))
        )

        # This will act as a glorified embedding layer for the action
        # Beware, do not add a layer norm here. It will cause the action to be normalized to 0
        # when len(bins) is 1.
        self.action_projection = nn.Sequential(
            nn.Linear(len(bins), d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        # self.center_indices = get_center_patch_indices(
        #     image_height, patch_height, image_width, patch_width, device
        # )

        params = sum(p.numel() for p in self.parameters())
        logger.debug(f"LatentActionVQVAE initialized. Num params: {params}")
        self.apply(init_weights)
        if zero_init_output_head:
            self.zero_init_output_head()

    def zero_init_output_head(self) -> None:
        """Make the decoder initially emit zero-valued residuals."""
        self.patch_to_pixels.zero_init_output()

    def decode(self, video: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        # Trim off the final image from the video for decoder input
        # Patches for frames 0 to T-2

        # Now, we need to decode. The decoder aims to predict frame T-1 using frames 0 to T-2 and the quantized_action.
        # decoded = self.decoder(patch_video_for_decoder)

        # Shape is (B, N_images_in_video - 1, num_patches, d_model)
        # logger.debug(
        #     f"decoded shape: {decoded.shape}"
        # )  # Decoder outputs a single frame (B, C, H, W)
        patched_video_for_decoder = self.embed_image_patches_decoder(video[:, :-1, :, :])

        logger.debug(f"patched_video_for_decoder shape: {patched_video_for_decoder.shape}")

        action_projected = self.action_projection(quantized)
        logger.debug(f"action_projected shape: {action_projected.shape}")

        x = patched_video_for_decoder + action_projected.unsqueeze(2)
        logger.debug(f"x shape: {x.shape}")

        x = self.decoder_transformer(x)
        logger.debug(f"x shape after decoder_transformer: {x.shape}")

        x = self.patch_to_pixels(x)
        logger.debug(f"x shape after patch_to_pixels: {x.shape}")

        return x

    def encode_with_fsq(self, video: torch.Tensor) -> torch.Tensor:
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

        # mean_pooled_patches = x_encoded_full.mean(dim=2)

        # Pair frame t with frame t+1, then concatenate their pooled embeddings.
        # Have explored residuals in the past but they are near 0 at times. Better to pass in both.
        first_images = x_encoded_full[:, :-1, :, :]
        target_images = x_encoded_full[:, 1:, :, :]
        residuals = target_images - first_images
        residuals = residuals.mean(dim=2)

        # (b, t, p, d) -> (b, t, p, 2d)
        # x_encoded_concat = torch.cat((first_images, target_images), dim=-1)

        # Reshape to (b, t, p, 2d) -> (b, t, p*2*d)
        # x_encoded_concat = rearrange(x_encoded_concat, "b t p d -> b t (p d)", d=self.d_model * 2)

        # New shape is 
        x_encoded_mean = self.action_head(self.pre_action_norm(residuals))

        logger.debug(f"x_encoded_mean shape: {x_encoded_mean.shape}")
        quantized = self.quantizer(x_encoded_mean.float())

        # quantized, indices, commitment_loss = self.quantizer(action_continuous)
        logger.debug(f"quantized shape: {quantized.shape}")
        # logger.debug(f"quantized values: {quantized}")
        return quantized

    def project_quantized_actions_fsq(self, quantized: torch.Tensor, patched_video_from_embedder: torch.Tensor) -> torch.Tensor:
        # Project the quantized action to the d_model space
        action_projected = self.action_projection(quantized)
        logger.debug(f"action_projected shape: {action_projected.shape}")

        return patched_video_from_embedder[:, :-1, :, :] + action_projected.unsqueeze(2)

    def project_quantized_actions_nsvq(self, quantized: torch.Tensor, patched_video_from_embedder: torch.Tensor) -> torch.Tensor:
        # quantized, indices, commitment_loss = self.quantizer(action_continuous)
        logger.debug(f"quantized shape: {quantized.shape}")

        actions = rearrange(
            quantized,
            "(b t) p d -> b t p d",
            b=patched_video_from_embedder.shape[0],
            t=patched_video_from_embedder.shape[1] - 1,
        )

        # Trim off the final image from the video for decoder input
        # Patches for frames 0 to T-2
        return patched_video_from_embedder[:, :-1, :, :] + actions

    def encode_with_nsvq(self, video: torch.Tensor) -> torch.Tensor:
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

        # We now have a sequnce of (B, num_frames, num_patches, d_model)
        # We want to quantize frames (0, 1), (1, 2), etc.
        # Quantizer expects first image, target image, each of shape (B, num_patches, d_model)
        # So, we need to reshape x_encoded_full to (B * (num_frames - 1), num_patches, d_model)
        # 1) Form per-batch consecutive pairs along time
        first_images = x_encoded_full[:, :-1, :, :]  # (B, T-1, P, D) → frames 0..T-1
        target_images = x_encoded_full[:, 1:, :, :]  # (B, T-1, P, D) → frames 1..T

        # 2) Now flatten (B, T-1) into a single batch dimension
        first_images = rearrange(
            first_images, "b t p d -> (b t) p d"
        )  # (B*(T-1), P, D)
        target_images = rearrange(
            target_images, "b t p d -> (b t) p d"
        )  # (B*(T-1), P, D)

        quantized = self.quantizer(first_images.float(), target_images.float())

        return quantized

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        # Return the quantized values and the patched video from the embedder
        if self.quantizer_type == "nsvq":
            return self.encode_with_nsvq(video)
        elif self.quantizer_type == "fsq":
            return self.encode_with_fsq(video)
        else:
            raise ValueError(f"Invalid quantizer type: {self.quantizer_type}")

    def get_action_sequence(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantized_value_to_codes(quantized.float())

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        actions = self.encode(video)
        return self.decode(video, actions)
