import logging

import torch
import torch.nn as nn

from latent_action_model.model import LatentActionVQVAE
from video_tokenization.model import VideoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DynamicsModel(nn.Module):
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
        action_model: LatentActionVQVAE,
        tokenizer: VideoTokenizer,
        device: torch.device,
    ):
        super(DynamicsModel, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.action_model = action_model
        self.tokenizer = tokenizer

        self.tokenizer.eval().to(device)
        self.action_model.eval().to(device)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, channels, image_height, image_width)

        # First, encode the video into a sequence of patches
        # We don't need the last frame for the tokenizer or the action model
        x = x[:, :-1, :, :]
        targets = x[:, 1:, :, :]
        with torch.no_grad():
            encoded_video = self.tokenizer.encode(x)
            targets = self.tokenizer.encode(targets)
            target_codes = self.tokenizer.quantized_value_to_codes(targets)

        logger.debug(f"encoded_video shape: {encoded_video.shape}")
        logger.debug(f"target_codes shape: {target_codes.shape}")

        # Next, grab the action sequence from the action model (original video, not encoded)
        action_video = self.action_model.encode(x)
        logger.debug(f"action_video shape: {action_video.shape}")

        # This will be of shape (batch_size, num_images_in_video - 1, num_patches, d_model)
        # Represents the action to go from frame i to frame i+1 for each frame in the video
        # TODO: Experiment with other ways to combine the embedding and the action video
        embedding = encoded_video + action_video

        # Now, we need to generate our prediction for the next frame
