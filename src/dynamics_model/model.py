from einops.einops import rearrange
import torch
import torch.nn as nn
from torch.distributions import uniform

from latent_action_model.model import LatentActionVQVAE
from loss.loss_fns import reconstruction_loss
from torch_utilities.initialize import init_weights
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer
from video_tokenization.model import VideoTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        mask_ratio_lower_bound: float,
        mask_ratio_upper_bound: float,
        num_images_in_video: int,
        num_heads: int,
        num_layers: int,
        d_model: int,
        tokenizer: VideoTokenizer,
        action_model: LatentActionVQVAE,
    ):
        super(DynamicsModel, self).__init__()
        self.mask_ratio_lower_bound = mask_ratio_lower_bound
        self.mask_ratio_upper_bound = mask_ratio_upper_bound
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_ratio_distribution = uniform.Uniform(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )

        self.tokenizer_vocab_size = tokenizer.get_vocab_size()
        self.action_vocab_size = action_model.action_vocab_size

        self.tokenizer = tokenizer
        self.action_model = action_model

        # python -m scripts.video_tokenizer.train --frames_dir pokemon_frames/pokemon_emerald --num_images_in_video 5 --batch_size 16 --save_dir dynamics_model_base --bins 8 8 6 5 --use_s3 --dataset_train_key pokemon_emerald_train_0_9_5_frames.json --checkpoint_dir dynamics_model_base --patch_size 4 --image_size 128 --num_epochs 20 --gradient_clipping 1.0 --tokenizer_checkpoint_path fsq_tokenizer_2k_128_4_512_8_heads_4_layers

        # +1 for the mask token
        self.tokenizer_embedding = nn.Embedding(self.tokenizer_vocab_size + 1, d_model)
        self.action_embedding = nn.Embedding(self.action_vocab_size + 1, d_model)
        self.decoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.vocab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, self.tokenizer.get_vocab_size()),
        )
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.action_loss_fn = reconstruction_loss

        init_weights(self)

    def forward(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # Return our predictions (batch_size, num_images_in_video, num_patches, vocab_size)
        # And the masked targets

        # x is of shape (batch_size, num_images_in_video, num_patches)
        with torch.no_grad():
            # x is of shape (batch_size, num_images_in_video, num_patches)
            targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video)
            ).long()

        batch_size, num_images_in_video, num_patches = targets.shape
        mask = (
            torch.rand(batch_size, num_images_in_video - 1, num_patches, device=targets.device)
            < self.mask_ratio_distribution.sample()
        )  # Boolean mask: True = masked position

        # Save original targets for loss computation before masking
        original_targets = targets.clone()

        targets[:, 1:, :] = self.tokenizer.mask_codebook_tokens(targets[:, 1:, :].long(), mask.long())

        # TODO: Should this also use the mask?
        # (batch_size, num_images_in_video - 1, num_patches, d_model)
        action_video_encoded, patched_video_from_action_embedder = self.action_model.encode(video)

        action_tokens = self.action_model.get_action_sequence(
            action_video_encoded
        )

        # targets is of shape (batch_size, num_images_in_video, num_patches)
        x = self.tokenizer_embedding(targets.long())
        action_embeddings = self.action_embedding(action_tokens.long())
        logger.info(f"action_embeddings shape: {action_embeddings.shape}")
        logger.info(f"x shape: {x.shape}")

        # Unsqueeze to add to each patch in the sequence
        x[:, :-1, :] += action_embeddings.unsqueeze(2)

        x = self.decoder(x)
        x = self.vocab_head(x)

        reconstructed_action_video = self.action_model.decode(action_video_encoded, patched_video_from_action_embedder)
        action_loss = self.action_loss_fn(
            video[:, 1:, :, :], reconstructed_action_video
        )

        # Use original targets for loss; ignore unmasked positions
        original_targets[:, 1:, :][~mask] = torch.tensor(-100, dtype=torch.long, device=original_targets.device)  # ~mask is proper boolean NOT here

        # View both as 3d tensors for the loss function
        logger.info(f"x shape: {x.shape}, targets shape: {targets.shape}")
        predicted_tokens = rearrange(x[:, 1:, :, :], "b t p d -> b (t p) d")
        target_tokens = rearrange(original_targets[:, 1:, :], "b t p -> b (t p)")
        logger.info(f"predicted_tokens shape: {predicted_tokens.shape}, target_tokens shape: {target_tokens.shape}")
        token_loss = self.token_loss_fn(predicted_tokens.transpose(1, 2), target_tokens)

        return x, token_loss, action_loss
