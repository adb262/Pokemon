import logging
import os

import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_video_to_images(input_video: torch.Tensor) -> list[list[Image.Image]]:
    """
    Converts a batch of videos in torch.Tensor format (B, T, C, H, W) or (B, T, H, W, C)
    into a nested list of PIL Images. Handles normalization and dtype conversion.

    Returns:
        images: list of list of PIL Images, shape [batch][time]
    """
    import numpy as np

    # If tensor is on GPU, move to CPU and detach
    video_tensor = input_video.detach().cpu()

    # Check shape and permute to (B, T, H, W, C)
    if video_tensor.dim() == 5:  # (B, T, C, H, W) or (B, T, H, W, C)
        if video_tensor.shape[2] == 3:  # likely (B, T, C, H, W)
            # Permute to (B, T, H, W, C)
            video_tensor = video_tensor.permute(0, 1, 3, 4, 2)
        elif video_tensor.shape[-1] == 3:  # already (B, T, H, W, C)
            pass
        else:
            raise ValueError(
                f"Unexpected channel dimension in input_video with shape {video_tensor.shape}"
            )
    else:
        raise ValueError(
            f"Expected 5D input [B, T, C, H, W] or [B, T, H, W, C], got {video_tensor.shape}"
        )

    # If the tensor is float, assume [0,1] and convert to uint8
    if video_tensor.dtype in [torch.float, torch.float32, torch.float64]:
        video_tensor = torch.clamp(video_tensor, 0, 1)  # clamp out-of-range values
        video_tensor = (video_tensor * 255).to(torch.uint8)
    else:
        video_tensor = video_tensor.to(torch.uint8)

    videos = video_tensor.numpy()
    images: list[list[Image.Image]] = []
    for video in videos:
        # video shape: (T, H, W, C)
        video_images: list[Image.Image] = []
        logger.info(f"Video shape: {video.shape}")
        for frame_idx in range(video.shape[0]):
            frame = video[frame_idx]
            # Defensive: If the mode is not correct, force RGB for 3-channels
            if frame.shape[-1] == 3:
                image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
            elif frame.shape[-1] == 1:
                image = Image.fromarray(frame[..., 0].astype(np.uint8), mode="L")
            else:
                raise ValueError(
                    f"Frame has unexpected channel count: {frame.shape[-1]}"
                )
            video_images.append(image)
        logger.info(f"Video images shape: {len(video_images)}")
        images.append(video_images)
    return images


def save_comparison_images_next_frame(
    predicted_videos: list[list[Image.Image]],
    predicted_actions: list[list[float]],
    expected_videos: list[list[Image.Image]],
    file_prefix: str,
):
    # For each video in the batch, save a single plot showing:
    # row 0: original frame
    # row 1: expected next frame
    # row 2: predicted next frame
    # with columns corresponding to time steps (from frame 1 onward).
    import matplotlib.pyplot as plt

    os.makedirs(file_prefix, exist_ok=True)

    for i, predicted_video in enumerate(predicted_videos):
        num_frames = len(predicted_video)

        fig, axs = plt.subplots(
            3,
            num_frames,
            figsize=(max(4, num_frames * 2.5), 3 * 2.5),
        )

        # If there's only one column, axs will be 1D; make it 2D for uniform indexing
        if num_frames - 1 == 1:
            import numpy as np

            axs = np.expand_dims(axs, axis=1)

        if len(expected_videos[i]) - num_frames != 1:
            raise ValueError(
                f"Expected {len(expected_videos[i])} frames, got {num_frames}"
            )

        for col in range(num_frames):
            # The 'original frame' is frame j-1
            original_image = expected_videos[i][col]
            # The 'expected next frame' is frame j
            expected_image = expected_videos[i][col + 1]
            # The 'predicted next frame' is also frame j (in predicted_videos)
            predicted_image = predicted_video[col]

            # Row 0: original, Row 1: expected, Row 2: predicted
            axs[0, col].imshow(original_image)
            axs[0, col].axis("off")
            if col == 0:
                axs[0, col].set_ylabel("Original", fontsize=10)

            axs[1, col].imshow(expected_image)
            axs[1, col].axis("off")
            if col == 0:
                axs[1, col].set_ylabel("Expected Next", fontsize=10)

            axs[2, col].imshow(predicted_image)
            axs[2, col].axis("off")
            if col == 0:
                axs[2, col].set_ylabel("Predicted Next", fontsize=10)

            axs[0, col].set_title(
                f"t={col} action={predicted_actions[i][col]}", fontsize=9
            )

        plt.tight_layout()
        plt.savefig(f"{file_prefix}/sample_{i}_next_frame_comparison.png")
        plt.close(fig)


def save_comparison_images(
    predicted_videos: list[list[Image.Image]],
    expected_videos: list[list[Image.Image]],
    file_prefix: str,
):
    # Save a single plot showing: original frame, predicted frame
    # The predicted frame should be the same as the expected frame
    import matplotlib.pyplot as plt

    # Put all frames in a single grid (rows: samples, columns: frames; each cell: [original, predicted] subcolumns)
    num_samples = len(predicted_videos)
    num_frames = len(predicted_videos[0]) if num_samples > 0 else 0
    import numpy as np

    fig, axs = plt.subplots(
        num_samples, num_frames * 2, figsize=(num_frames * 2.5 * 2, num_samples * 2.5)
    )

    # Handle 1D axes (if only one sample)
    if num_samples == 1:
        axs = np.expand_dims(axs, 0)
    if num_frames == 1:
        axs = np.expand_dims(axs, 1)

    for i, predicted_video in enumerate(predicted_videos):
        for j in range(num_frames):
            # The 'original frame' is frame j
            original_image = expected_videos[i][j]
            # The 'predicted frame' is also frame j (in predicted_videos)
            predicted_image = predicted_video[j]

            axs[i, j * 2].imshow(original_image)
            axs[i, j * 2].set_title(f"Orig S{i}F{j}")
            axs[i, j * 2].axis("off")

            axs[i, j * 2 + 1].imshow(predicted_image)
            axs[i, j * 2 + 1].set_title(f"Pred S{i}F{j}")
            axs[i, j * 2 + 1].axis("off")

    plt.tight_layout()
    os.makedirs(file_prefix, exist_ok=True)
    plt.savefig(f"{file_prefix}/comparison_grid.png")
    plt.close(fig)
