# python -m scripts.mask_git.train --tokenizer_checkpoint_path checkpoints/tokenizer.pt --frames_dir pokemon --num_images_in_video 4 --batch_size 4
import logging

from dynamics_model.training_args import DynamicsModelTrainingConfig
from latent_action_model.model import LatentActionVQVAE
from transformers.mask_git import DynamicsModel
from video_tokenization.model import VideoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def create_dynamics_model(
    config: DynamicsModelTrainingConfig,
    tokenizer: VideoTokenizer,
    action_model: LatentActionVQVAE,
) -> DynamicsModel:
    """Create and initialize the DynamicsModel model"""
    model = DynamicsModel(
        mask_ratio_lower_bound=config.mask_ratio_lower_bound,
        mask_ratio_upper_bound=config.mask_ratio_upper_bound,
        num_images_in_video=config.num_images_in_video,
        num_heads=config.dynamics_num_heads,
        num_layers=config.dynamics_num_transformer_layers,
        d_model=config.dynamics_d_model,
        tokenizer=tokenizer,
        action_model=action_model,
    )
    return model
