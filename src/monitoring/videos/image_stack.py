"""Utilities for stacking and combining images."""

from pathlib import Path
from typing import Literal, Sequence

from PIL import Image


def stack_images_vertically(
    images: Sequence[Image.Image],
    *,
    pad: int = 0,
    bg: tuple[int, int, int] = (0, 0, 0),
    align: Literal["left", "center", "right"] = "left",
) -> Image.Image:
    """
    Stack multiple PIL images vertically into a single image.

    Args:
        images: Sequence of PIL Images to stack.
        pad: Padding in pixels between images.
        bg: Background color as RGB tuple.
        align: Horizontal alignment of images ("left", "center", or "right").

    Returns:
        A single PIL Image with all input images stacked vertically.
    """
    if not images:
        raise ValueError("Cannot stack empty sequence of images")

    # Convert all images to RGB mode for consistency
    rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    max_width = max(img.width for img in rgb_images)
    total_height = sum(img.height for img in rgb_images) + pad * (len(rgb_images) - 1)

    combined = Image.new("RGB", (max_width, total_height), color=bg)

    y_offset = 0
    for img in rgb_images:
        if align == "left":
            x_offset = 0
        elif align == "center":
            x_offset = (max_width - img.width) // 2
        else:  # right
            x_offset = max_width - img.width

        combined.paste(img, (x_offset, y_offset))
        y_offset += img.height + pad

    return combined


def stack_image_paths_vertically(
    paths: Sequence[str | Path],
    *,
    pad: int = 0,
    bg: tuple[int, int, int] = (0, 0, 0),
    align: Literal["left", "center", "right"] = "left",
) -> Image.Image:
    """
    Load images from file paths and stack them vertically.

    Args:
        paths: Sequence of file paths to images.
        pad: Padding in pixels between images.
        bg: Background color as RGB tuple.
        align: Horizontal alignment of images ("left", "center", or "right").

    Returns:
        A single PIL Image with all input images stacked vertically.
    """
    if not paths:
        raise ValueError("Cannot stack empty sequence of paths")

    images: list[Image.Image] = []
    for path in paths:
        img = Image.open(path)
        # Load fully into memory so file handle is released
        img.load()
        images.append(img)

    return stack_images_vertically(images, pad=pad, bg=bg, align=align)

