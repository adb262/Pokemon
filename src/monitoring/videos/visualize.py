"""Video visualization utilities for converting tensors to images and saving comparison grids."""

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
        logger.debug(f"Video shape: {video.shape}")
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
        logger.debug(f"Video images shape: {len(video_images)}")
        images.append(video_images)
    return images


def save_comparison_images_next_frame(
    predicted_videos: list[list[Image.Image]],
    predicted_actions: list[list[float]],
    expected_videos: list[list[Image.Image]],
    file_prefix: str,
    file_suffix: str = "next_frame_comparison_grid.png",
):
    """
    Save a comparison grid showing original frame, expected next frame, and predicted next frame.

    Each sample gets 3 rows: original, expected next, predicted next.
    Columns correspond to time steps.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(file_prefix, exist_ok=True)

    num_samples = len(predicted_videos)
    if num_samples == 0:
        return

    num_frames = len(predicted_videos[0])

    # Sanity check all lengths match up
    for i in range(num_samples):
        if len(expected_videos[i]) - len(predicted_videos[i]) != 1:
            raise ValueError(
                f"Sample {i}: Expected {len(expected_videos[i])} frames, got {len(predicted_videos[i])}"
            )

    # Each sample gets 3 rows: original, expected next, predicted next
    total_rows = num_samples * 3
    fig, axs = plt.subplots(
        total_rows,
        num_frames,
        figsize=(max(8, num_frames * 2.2), total_rows * 2.0),
    )

    # If only one row or one column, axs might be 1D, expand as needed
    if total_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)

    for sample_idx in range(num_samples):
        predicted_video = predicted_videos[sample_idx]
        expected_video = expected_videos[sample_idx]
        actions = predicted_actions[sample_idx]
        row_offset = sample_idx * 3

        for col in range(num_frames):
            # The 'original frame' is frame j-1
            original_image = expected_video[col]
            # The 'expected next frame' is frame j
            expected_image = expected_video[col + 1]
            # The 'predicted next frame' is also frame j (in predicted_videos)
            predicted_image = predicted_video[col]

            # Row 0: original
            axs[row_offset + 0, col].imshow(original_image)
            axs[row_offset + 0, col].axis("off")
            if col == 0:
                axs[row_offset + 0, col].set_ylabel(
                    f"Sample {sample_idx}\nOriginal", fontsize=10
                )

            # Row 1: expected next
            axs[row_offset + 1, col].imshow(expected_image)
            axs[row_offset + 1, col].axis("off")
            if col == 0:
                axs[row_offset + 1, col].set_ylabel("Expected Next", fontsize=10)

            # Row 2: predicted next
            axs[row_offset + 2, col].imshow(predicted_image)
            axs[row_offset + 2, col].axis("off")
            if col == 0:
                axs[row_offset + 2, col].set_ylabel("Predicted Next", fontsize=10)

            # Set the title for the top row of the sample (original frame)
            axs[row_offset + 0, col].set_title(
                f"t={col}  action={actions[col]}", fontsize=9
            )

    plt.tight_layout()
    plt.savefig(f"{file_prefix}/{file_suffix}")
    plt.close(fig)


def save_rollout_comparison_grid(
    gt_videos: list[list[Image.Image]],
    predicted_videos: list[list[Image.Image]],
    predicted_actions: list[list[int]],
    output_dir: str,
    context_len: int,
    file_suffix: str = "rollout_comparison_grid.png",
):
    """Save a comparison grid for a dynamics-model rollout eval.

    Each sample produces 2 rows × 2T columns:
      Row 0 (GT):        gt[0..2T-1]
      Row 1 (Predicted): gt[0..T-1] then pred_0..pred_{T-1}

    A vertical divider is drawn between the context region (cols 0..T-1)
    and the rollout region (cols T..2T-1).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(gt_videos)
    if num_samples == 0:
        return

    total_cols = len(gt_videos[0])

    total_rows = num_samples * 2
    fig, axs = plt.subplots(
        total_rows,
        total_cols,
        figsize=(max(8, total_cols * 2.2), total_rows * 2.0),
    )

    if total_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if total_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for sample_idx in range(num_samples):
        gt = gt_videos[sample_idx]
        pred = predicted_videos[sample_idx]
        actions = predicted_actions[sample_idx]
        row_offset = sample_idx * 2

        for col in range(total_cols):
            # Row 0: ground truth
            axs[row_offset, col].imshow(gt[col])
            axs[row_offset, col].axis("off")
            if col == 0:
                axs[row_offset, col].set_ylabel(
                    f"Sample {sample_idx}\nGT", fontsize=10
                )

            # Row 1: predicted (context region = raw GT, rollout region = model predictions)
            axs[row_offset + 1, col].imshow(pred[col])
            axs[row_offset + 1, col].axis("off")
            if col == 0:
                axs[row_offset + 1, col].set_ylabel("Predicted", fontsize=10)

            # Column titles on top row of each sample
            if col < context_len:
                title = f"t={col}"
            else:
                action_idx = col - context_len
                action_val = actions[action_idx] if action_idx < len(actions) else "?"
                title = f"t={col}  a={action_val}"
            axs[row_offset, col].set_title(title, fontsize=9)

        # Draw vertical divider at the context/rollout boundary
        for row in range(2):
            r = row_offset + row
            if context_len - 1 < total_cols:
                ax_left = axs[r, context_len - 1]
                for spine in ax_left.spines.values():
                    spine.set_visible(False)
                ax_left.spines["right"].set_visible(True)
                ax_left.spines["right"].set_color("red")
                ax_left.spines["right"].set_linewidth(3)
            if context_len < total_cols:
                ax_right = axs[r, context_len]
                for spine in ax_right.spines.values():
                    spine.set_visible(False)
                ax_right.spines["left"].set_visible(True)
                ax_right.spines["left"].set_color("red")
                ax_right.spines["left"].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_suffix}")
    plt.close(fig)


def save_comparison_images(
    predicted_videos: list[list[Image.Image]],
    expected_videos: list[list[Image.Image]],
    output_path: str,
) -> str:
    """
    Save a comparison grid showing original and predicted frames side by side.

    Args:
        predicted_videos: List of predicted video frames per sample.
        expected_videos: List of expected video frames per sample.
        output_path: Full path where the comparison image will be saved.

    Returns:
        The output_path where the image was saved.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_samples = len(predicted_videos)
    num_frames = len(predicted_videos[0]) if num_samples > 0 else 0

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path

