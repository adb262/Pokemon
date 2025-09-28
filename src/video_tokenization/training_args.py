

from dataclasses import dataclass

import torch


@dataclass
class VideoTokenizerTrainingConfig:
    image_size: int = 400
    patch_size: int = 16
    batch_size: int = 16
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    num_epochs: int = 2
    device: str = 'mps' if torch.backends.mps.is_available() else 'cuda'
    frames_dir: str = 'pokemon_frames'
    log_interval: int = 10
    save_interval: int = 500