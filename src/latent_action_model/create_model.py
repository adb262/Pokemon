import logging

import torch

from dynamics_model.checkpoints import adapt_state_dict_to_model
from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.model import LatentActionVQVAE
from latent_action_model.training_args import VideoTrainingConfig

logger = logging.getLogger(__name__)


def _compile_action_model_for_checkpoint_load(model: LatentActionVQVAE) -> None:
    """Wrap the same submodules training compiles, so checkpoint keys align.

    The training script wraps ``model.encoder`` and ``model.decoder_transformer``
    with ``torch.compile``, which inserts ``._orig_mod.`` into those submodules'
    ``state_dict`` keys. Compiling here before ``load_state_dict`` makes the
    model's expected keys match checkpoints that were saved while compiled
    (e.g. ``checkpoint_epoch1_batch12749`` from
    ``dynamics_model_pong_w_tokenizer_v2_256_scheduled_opt_longer_eval_128_d_action``).
    Canonical (already-stripped) checkpoints still load correctly because
    ``adapt_state_dict_to_model`` re-inserts ``_orig_mod`` to match.
    """
    model.encoder = torch.compile(model.encoder, dynamic=True)  # type: ignore[assignment]
    model.decoder_transformer = torch.compile(  # type: ignore[assignment]
        model.decoder_transformer, dynamic=True
    )


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
        # zero_init_output_head=config.predict_action_residuals,
        zero_init_output_head=False,
    )

    if config.action_model_checkpoint_path:
        _compile_action_model_for_checkpoint_load(model)
        checkpoint = torch.load(config.action_model_checkpoint_path)
        state_dict = adapt_state_dict_to_model(
            checkpoint["model_state_dict"], model
        )
        model.load_state_dict(state_dict)
        logger.info(
            "Loaded action model weights from %s",
            config.action_model_checkpoint_path,
        )

    return model
