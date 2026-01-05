from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.model import LatentActionVQVAE
from latent_action_model.training_args import VideoTrainingConfig


def create_action_model(
    config: VideoTrainingConfig,
) -> LatentActionVQVAE:
    """Create and initialize the latent action model for co-training"""
    model = LatentActionVQVAE(
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
        use_temporal_transformer=True,
        use_spatial_transformer=True,
        quantizer_type="fsq",
        bins=config.bins,
    )
    return model


def create_action_model_from_dynamics_config(
    config: DynamicsModelTrainingConfig,
) -> LatentActionVQVAE:
    """Create and initialize the latent action model for co-training"""
    model = LatentActionVQVAE(
        channels=3,
        image_height=config.image_size,
        image_width=config.image_size,
        patch_height=config.patch_size,
        patch_width=config.patch_size,
        num_images_in_video=config.num_images_in_video,
        d_model=config.action_d_model,
        num_heads=config.action_num_heads,
        num_layers=config.action_num_transformer_layers,
        embedding_dim=config.action_latent_dim,
        use_temporal_transformer=True,
        use_spatial_transformer=True,
        quantizer_type="fsq",
        bins=config.action_bins,
    )
    return model
