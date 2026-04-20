"""Back-compat alias — prefer importing VideoWindowLoader directly."""

from data.data_loaders.video_window_loader import VideoWindowLoader

PokemonOpenWorldLoader = VideoWindowLoader

__all__ = ["PokemonOpenWorldLoader"]
