import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Cache:
    hits: int
    misses: int

    def __init__(self, max_size: int, cache_dir: str):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0
        if cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_cached_image_path(self, key: str) -> str:
        return key.replace("\\", "_")

    def get(self, key: str) -> Image.Image | None:
        path = os.path.join(self.cache_dir, self.get_cached_image_path(key))
        if os.path.exists(path):
            self.hits += 1
            return Image.open(path)

        self.misses += 1
        return None

    def set(self, key: str, value: Image.Image):
        value.save(f"{self.cache_dir}/{self.get_cached_image_path(key)}", "PNG")

    def print_stats(self):
        print(f"Cache hits: {self.hits}")
        print(f"Cache misses: {self.misses}")
