import torch


def get_center_patch_indices(
    image_height: int,
    patch_height: int,
    image_width: int,
    patch_width: int,
    device: torch.device,
) -> torch.Tensor:
    # Only look at the middle patches. this is where the action with the character is happening.
    # x_encoded_full has shape (B, T, P, D) where P = num_patches_h * num_patches_w.
    # We want to restrict P to just the central region of the patch grid (e.g. 4x4 center if the grid is 16x16).
    num_patches_h = image_height // patch_height
    num_patches_w = image_width // patch_width

    # Size of the central window along each spatial dimension.
    # For a 16x16 grid this becomes 4x4, as requested.
    window_h = max(1, num_patches_h // 4)
    window_w = max(1, num_patches_w // 4)

    start_h = (num_patches_h - window_h) // 2
    start_w = (num_patches_w - window_w) // 2
    end_h = start_h + window_h
    end_w = start_w + window_w

    rows = torch.arange(start_h, end_h, device=device)
    cols = torch.arange(start_w, end_w, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols)
    center_indices = (row_grid * num_patches_w + col_grid).reshape(-1)
    return center_indices
