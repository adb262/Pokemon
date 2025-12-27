from video_tokenization.tokenizer import VideoTokenizer
from video_tokenization.training_args import VideoTokenizerTrainingConfig


def create_model(config: VideoTokenizerTrainingConfig):
    """Create and initialize the VQVAE model"""
    model = VideoTokenizer(
        channels=3,
        image_height=config.image_size,
        image_width=config.image_size,
        patch_height=config.patch_size,
        patch_width=config.patch_size,
        num_images_in_video=config.num_images_in_video,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        embedding_dim=config.latent_dim,
        bins=config.bins,
    )

    return model
