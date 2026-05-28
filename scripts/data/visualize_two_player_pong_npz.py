#!/usr/bin/env python3
"""Render labeled previews from a two-player Pong `.npz` shard.

Example:
    python -m scripts.data.visualize_two_player_pong_npz \
        data/two_player_pong_smoke/train/chunk_000000.npz
"""

import os
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import tyro
from tyro.conf import Positional
from PIL import Image, ImageDraw


ACTION_NAMES: dict[int, str] = {
    0: "nothing",
    1: "up",
    2: "down",
}
RAW_ACTION_NAMES: dict[int, str] = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
}


@dataclass
class Args:
    npz_path: Annotated[str, Positional]
    output_path: str | None = None
    num_samples: int = 8
    start_index: int = 0
    start_frame: int = 0
    max_frames: int = 16
    label_band_height: int = 22
    show_raw_actions: bool = False
    show_residuals: bool = False
    residual_offset: int = 1
    residual_scale: float = 8.0
    upscale: int = 3


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 5:
        raise ValueError(f"Expected frames with 5 dims, got shape {frames.shape}")

    if frames.shape[-1] == 3:
        normalized = frames
    elif frames.shape[2] == 3:
        normalized = np.transpose(frames, (0, 1, 3, 4, 2))
    else:
        raise ValueError(
            "Expected frames in NTHWC or NTCHW layout with 3 channels, "
            f"got shape {frames.shape}"
        )

    if normalized.dtype != np.uint8:
        clipped = np.clip(normalized, 0, 255)
        normalized = clipped.astype(np.uint8)
    return normalized


def output_path_for_args(args: Args) -> str:
    if args.output_path is not None:
        return args.output_path
    base_path, _extension = os.path.splitext(args.npz_path)
    suffix = "residuals" if args.show_residuals else "labels"
    return f"{base_path}_{suffix}.png"


def make_residual_frames(
    frames: np.ndarray,
    residual_scale: float,
    residual_offset: int,
) -> np.ndarray:
    if residual_offset < 1:
        raise ValueError("residual_offset must be at least 1")
    if frames.shape[1] <= residual_offset:
        raise ValueError(
            f"Need more than {residual_offset} frames to render residuals"
        )
    residuals = np.abs(
        frames[:, residual_offset:].astype(np.int16)
        - frames[:, :-residual_offset].astype(np.int16)
    ).astype(np.float32)
    residuals *= residual_scale
    return np.clip(residuals, 0, 255).astype(np.uint8)


def validate_actions(dual_actions: np.ndarray, frames: np.ndarray) -> None:
    if dual_actions.ndim != 3 or dual_actions.shape[-1] != 2:
        raise ValueError(
            "Expected dual_actions with shape (N, T - 1, 2), "
            f"got {dual_actions.shape}"
        )
    expected_steps = frames.shape[1] - 1
    if (
        dual_actions.shape[0] != frames.shape[0]
        or dual_actions.shape[1] != expected_steps
    ):
        raise ValueError(
            "dual_actions must align with frames as (N, T - 1, 2): "
            f"frames={frames.shape}, dual_actions={dual_actions.shape}"
        )
    if np.any(dual_actions < 0) or np.any(dual_actions > 2):
        raise ValueError("dual_actions values must be in [0, 2]")


def action_text(
    dual_actions: np.ndarray,
    row_idx: int,
    frame_idx: int,
    raw_actions: np.ndarray | None,
    show_raw_actions: bool,
) -> str:
    if frame_idx >= dual_actions.shape[1]:
        return ""

    left_label = int(dual_actions[row_idx, frame_idx, 0])
    right_label = int(dual_actions[row_idx, frame_idx, 1])
    text = f"L:{ACTION_NAMES[left_label]} R:{ACTION_NAMES[right_label]}"

    if show_raw_actions and raw_actions is not None:
        left_raw = int(raw_actions[row_idx, frame_idx, 0])
        right_raw = int(raw_actions[row_idx, frame_idx, 1])
        text = (
            f"{text} "
            f"raw=({RAW_ACTION_NAMES.get(left_raw, str(left_raw))},"
            f"{RAW_ACTION_NAMES.get(right_raw, str(right_raw))})"
        )
    return text


def render_preview(
    frames: np.ndarray,
    dual_actions: np.ndarray,
    output_path: str,
    args: Args,
    raw_actions: np.ndarray | None,
) -> None:
    if frames.shape[0] == 0:
        raise ValueError("Shard contains no frames")

    start_index = max(args.start_index, 0)
    stop_index = min(start_index + args.num_samples, frames.shape[0])
    selected_frames = frames[start_index:stop_index]
    selected_actions = dual_actions[start_index:stop_index]
    selected_raw_actions = (
        raw_actions[start_index:stop_index] if raw_actions is not None else None
    )
    if selected_frames.shape[0] == 0:
        raise ValueError(
            f"No samples selected from {frames.shape[0]} available windows; "
            f"start_index={args.start_index}, num_samples={args.num_samples}"
        )
    start_frame = max(args.start_frame, 0)
    stop_frame = min(start_frame + args.max_frames, selected_frames.shape[1])
    if stop_frame <= start_frame:
        raise ValueError(
            f"No frames selected from {selected_frames.shape[1]} available frames; "
            f"start_frame={args.start_frame}, max_frames={args.max_frames}"
        )
    selected_frames = selected_frames[:, start_frame:stop_frame]
    selected_actions = selected_actions[:, start_frame : max(stop_frame - 1, start_frame)]
    if selected_raw_actions is not None:
        selected_raw_actions = selected_raw_actions[
            :,
            start_frame : max(stop_frame - 1, start_frame),
        ]

    display_frames = selected_frames
    if args.show_residuals:
        display_frames = make_residual_frames(
            selected_frames,
            args.residual_scale,
            args.residual_offset,
        )

    rows, columns, frame_height, frame_width, _channels = display_frames.shape
    scale = max(args.upscale, 1)
    cell_width = frame_width * scale
    cell_height = frame_height * scale
    row_height = cell_height + args.label_band_height
    sheet = Image.new("RGB", (columns * cell_width, rows * row_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for row_idx in range(rows):
        row_top = row_idx * row_height
        sample_idx = start_index + row_idx
        draw.text(
            (2, row_top + 2),
            f"sample {sample_idx}, frames {start_frame}:{stop_frame}"
            + (f", residual +{args.residual_offset}" if args.show_residuals else ""),
            fill=(255, 255, 0),
        )

        for frame_idx in range(columns):
            frame_image = Image.fromarray(display_frames[row_idx, frame_idx])
            if scale != 1:
                frame_image = frame_image.resize(
                    (cell_width, cell_height),
                    resample=Image.Resampling.NEAREST,
                )
            x_pos = frame_idx * cell_width
            sheet.paste(frame_image, (x_pos, row_top + args.label_band_height))

            label = action_text(
                selected_actions,
                row_idx,
                frame_idx,
                selected_raw_actions,
                args.show_raw_actions,
            )
            if label:
                draw.text((x_pos + 2, row_top + 11), label, fill=(255, 255, 255))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = tyro.cli(Args)
    with np.load(args.npz_path) as shard:
        if "frames" not in shard.files:
            raise KeyError(f"{args.npz_path} is missing required array 'frames'")
        if "dual_actions" not in shard.files:
            raise KeyError(f"{args.npz_path} is missing required array 'dual_actions'")

        frames = normalize_frames(shard["frames"])
        dual_actions = shard["dual_actions"].astype(np.int64)
        raw_actions = None
        if "raw_actions" in shard.files:
            raw_actions = shard["raw_actions"].astype(np.int64)

    validate_actions(dual_actions, frames)
    output_path = output_path_for_args(args)
    render_preview(frames, dual_actions, output_path, args, raw_actions)
    print(f"Wrote labeled preview to {output_path}")


if __name__ == "__main__":
    main()
