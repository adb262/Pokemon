import os
from typing import Counter

import numpy as np
from PIL import Image


def get_counts_of_pixels(arr: np.ndarray) -> Counter:
    # Get the Counter of connected same RGB pixels in the array
    # These will be rounded to the nearest 10 for sake of ignoring noise
    counter = Counter()
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            key = [int(round(val, -1)) for val in arr[x, y].tolist()]
            counter[str(key)] += 1
    return counter


def is_frame_valid(image: Image.Image) -> bool:
    """
    Returns True if the frame is likely to be in a navigable environment.
    Heuristics:
    - No text box at the bottom (e.g., dialog or menu)
    - Player sprite is visible in the center
    - Not in battle/menu/cutscene/dialog
    """
    img = image.convert("RGB")
    arr = np.array(img)

    # Trim off 5% of the right and left edges
    h, w, _ = arr.shape
    arr = arr[:, int(w * 0.05) : w - int(w * 0.05), :]

    h, w, _ = arr.shape

    # 1. Check for text box at the bottom (common in Pokemon games)
    # Assume text box is a solid color (often white/gray/black) at the bottom ~20% of the screen
    bottom_pct = 0.20
    bottom = arr[int(h * (1 - bottom_pct)) :, :, :]

    # Look for pools of connected pixels
    # if there is a large pool of pixels, it's likely a text box
    pixel_counts = get_counts_of_pixels(bottom)

    most_common_pixel_count = sum([x[1] for x in pixel_counts.most_common(3)])
    most_common_pixel = pixel_counts.most_common(1)[0][0]
    total_pixel_count = sum(pixel_counts.values())
    threshold = 0.4 if most_common_pixel == "[255, 255, 255]" else 0.2

    print(
        most_common_pixel_count,
        total_pixel_count,
        most_common_pixel_count / total_pixel_count,
    )
    if most_common_pixel_count / total_pixel_count > threshold:
        return False

    # 2. Check for player sprite in the center
    # Assume player sprite is in the center 20% x 20% region
    center_pct = 0.20
    ch, cw = int(h * center_pct), int(w * center_pct)
    center = arr[
        h // 2 - ch // 2 : h // 2 + ch // 2, w // 2 - cw // 2 : w // 2 + cw // 2, :
    ]
    # If center is very uniform, likely not a sprite (e.g., menu, cutscene)
    center_std = np.std(center, axis=(0, 1))
    if np.mean(center_std) < 10:  # threshold may need tuning
        return False

    # If all checks passed, assume frame is valid
    return True


def get_frame_similarity(frame1: Image.Image, frame2: Image.Image):
    return np.mean(np.abs(np.array(frame1) - np.array(frame2))) / 255.0


def filter_frame_sequence(frame_sequence: list[str]) -> list[str]:
    frames = [Image.open(frame) for frame in frame_sequence]
    valid_frames = [
        (i, frame) for i, frame in enumerate(frames) if is_frame_valid(frame)
    ]
    if len(valid_frames) < 2:
        return []

    valid_frame_paths = []

    for i in range(len(valid_frames) - 1):
        curr_frame = valid_frames[i][1]
        next_frame = valid_frames[i + 1][1]
        similarity = get_frame_similarity(curr_frame, next_frame)
        if similarity >= 0.5 and similarity <= 0.99:
            if len(valid_frame_paths) == 0:
                valid_frame_paths.append(valid_frames[i][0])
            valid_frame_paths.append(valid_frames[i + 1][0])

    return valid_frame_paths


if __name__ == "__main__":
    for frame in os.listdir("cache"):
        with open(os.path.join("cache", frame), "rb") as f:
            image = Image.open(f)

            if is_valid := is_frame_valid(image):
                # visualize the frame
                image.show()
                print(f"Frame {frame} is valid: {is_valid}")
                import pdb

                pdb.set_trace()
