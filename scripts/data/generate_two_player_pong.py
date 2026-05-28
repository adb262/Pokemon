#!/usr/bin/env python3
"""Collect isolated two-player Pong data with dual paddle action labels.

Example smoke run:
    python -m scripts.data.generate_two_player_pong \
        --output-dir data/two_player_pong_smoke \
        --num-windows-train 8 \
        --num-windows-val 4 \
        --num-windows-test 4 \
        --windows-per-file 4 \
        --window-size 160
"""

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
import tyro
from pettingzoo.atari import pong_v3
from PIL import Image, ImageDraw
from tqdm import tqdm


logger = logging.getLogger(__name__)

AGENT_ORDER = ("first_0", "second_0")
PADDLE_ORDER = ("left", "right")
AGENT_TO_PADDLE: dict[str, str] = {
    "first_0": "right",
    "second_0": "left",
}

NORMALIZED_ACTION_TO_NAME: dict[int, str] = {
    0: "nothing",
    1: "up",
    2: "down",
}
NORMALIZED_ACTION_TO_RAW: dict[int, int] = {
    0: 0,  # NOOP
    1: 2,  # UP
    2: 5,  # DOWN
}
RAW_ACTION_TO_NAME: dict[int, str] = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
}
FIRE_ACTION = 1
LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


@dataclass
class Args:
    output_dir: str = "data/two_player_pong"
    num_windows_train: int = 10_000
    num_windows_val: int = 1_000
    num_windows_test: int = 1_000
    window_size: int = 160
    window_stride: int = 1
    windows_per_file: int = 1024
    frame_size: int = 84
    max_episode_steps: int = 1_000
    max_episodes_per_split: int = 10_000
    min_gameplay_step: int = 30
    require_ball_visible: bool = True
    min_ball_visible_fraction: float = 0.25
    ball_threshold: int = 100
    ball_min_pixels: int = 1
    ball_max_component_pixels: int = 20
    auto_fire_when_ball_missing: bool = True
    seed: int = 0
    policy: Literal["random", "sticky_random", "tracking"] = "random"
    sticky_action_prob: float = 0.9
    tracking_deadzone: int = 3
    serve_fire_steps: int = 1
    compress: bool = True
    preview_samples: int = 16
    overwrite: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@dataclass
class SplitBuffers:
    frames: list[np.ndarray]
    dual_actions: list[np.ndarray]
    raw_actions: list[np.ndarray]
    episode_ids: list[int]
    start_offsets: list[int]


@dataclass
class Episode:
    frames: list[np.ndarray]
    dual_actions: list[tuple[int, int]]
    raw_actions: list[tuple[int, int]]
    episode_id: int


def _reset_env(env, seed: int) -> dict[str, np.ndarray]:
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        observations, _infos = reset_result
        return observations
    return reset_result


def _step_env(
    env, action_by_agent: dict[str, int]
) -> tuple[dict[str, np.ndarray], bool]:
    step_result = env.step(action_by_agent)
    if len(step_result) == 5:
        observations, _rewards, terminations, truncations, _infos = step_result
        done = all(
            bool(terminations[agent]) or bool(truncations[agent])
            for agent in terminations
        )
        return observations, done

    observations, _rewards, dones, _infos = step_result
    done = all(bool(done_value) for done_value in dones.values())
    return observations, done


def preprocess_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    rgb_frame = np.asarray(frame, dtype=np.uint8)
    if rgb_frame.ndim != 3 or rgb_frame.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB frame with shape (H, W, 3), got {rgb_frame.shape}"
        )
    grayscale = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(
        grayscale,
        (frame_size, frame_size),
        interpolation=cv2.INTER_AREA,
    )
    return np.repeat(resized[:, :, None], 3, axis=-1).astype(np.uint8)


def has_ball_visible(frame: np.ndarray, args: Args) -> bool:
    """Detect a small bright ball in the downsampled grayscale Pong playfield."""
    grayscale = frame[:, :, 0]
    height, width = grayscale.shape
    y_min = max(height // 8, 1)
    y_max = max(height - 2, y_min + 1)
    x_min = max(width // 7, 1)
    x_max = min(width - width // 7, width)
    playfield = grayscale[y_min:y_max, x_min:x_max].copy()

    center_x = width // 2 - x_min
    center_margin = max(width // 42, 1)
    left = max(center_x - center_margin, 0)
    right = min(center_x + center_margin + 1, playfield.shape[1])
    playfield[:, left:right] = 0

    bright_mask = (playfield >= args.ball_threshold).astype(np.uint8)
    if int(bright_mask.sum()) < args.ball_min_pixels:
        return False

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        bright_mask,
        connectivity=8,
    )
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        width_px = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height_px = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if (
            args.ball_min_pixels <= area <= args.ball_max_component_pixels
            and width_px <= 6
            and height_px <= 6
        ):
            return True
    return False


def random_labels(rng: np.random.Generator) -> tuple[int, int]:
    labels = rng.integers(0, 3, size=2, dtype=np.int64)
    return int(labels[0]), int(labels[1])


def sticky_random_labels(
    rng: np.random.Generator,
    previous_labels: tuple[int, int],
    sticky_action_prob: float,
) -> tuple[int, int]:
    if rng.random() < sticky_action_prob:
        return previous_labels
    return random_labels(rng)


def _bright_pixel_y_center(frame: np.ndarray, x_min: int, x_max: int) -> float | None:
    grayscale = frame.mean(axis=-1)
    region = grayscale[:, x_min:x_max]
    y_positions, _x_positions = np.where(region > 80.0)
    if y_positions.size == 0:
        return None
    return float(y_positions.mean())


def tracking_labels(frame: np.ndarray, deadzone: int) -> tuple[int, int]:
    height, width, _channels = frame.shape
    left_paddle_y = _bright_pixel_y_center(frame, 0, max(width // 4, 1))
    right_paddle_y = _bright_pixel_y_center(
        frame,
        min((3 * width) // 4, width - 1),
        width,
    )
    ball_y = _bright_pixel_y_center(
        frame,
        width // 4,
        max((3 * width) // 4, width // 4 + 1),
    )
    if ball_y is None:
        ball_y = height / 2.0

    def label_for_paddle(paddle_y: float | None) -> int:
        if paddle_y is None:
            return 0
        if ball_y < paddle_y - deadzone:
            return 1
        if ball_y > paddle_y + deadzone:
            return 2
        return 0

    return label_for_paddle(left_paddle_y), label_for_paddle(right_paddle_y)


def choose_dual_labels(
    args: Args,
    rng: np.random.Generator,
    frame: np.ndarray,
    previous_labels: tuple[int, int],
) -> tuple[int, int]:
    if args.policy == "random":
        return random_labels(rng)
    if args.policy == "sticky_random":
        return sticky_random_labels(rng, previous_labels, args.sticky_action_prob)
    if args.policy == "tracking":
        return tracking_labels(frame, args.tracking_deadzone)
    raise ValueError(f"Unknown policy: {args.policy}")


def labels_to_raw_actions(
    labels: tuple[int, int],
    *,
    force_fire: bool,
) -> tuple[int, int]:
    if force_fire:
        return FIRE_ACTION, FIRE_ACTION
    return (
        NORMALIZED_ACTION_TO_RAW[labels[0]],
        NORMALIZED_ACTION_TO_RAW[labels[1]],
    )


def collect_episode(
    env,
    args: Args,
    rng: np.random.Generator,
    episode_id: int,
) -> Episode:
    episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    observations = _reset_env(env, episode_seed)
    current_frame = preprocess_frame(observations[AGENT_ORDER[0]], args.frame_size)
    frames = [current_frame]
    dual_actions: list[tuple[int, int]] = []
    raw_actions: list[tuple[int, int]] = []
    previous_labels = (0, 0)

    for step_idx in range(args.max_episode_steps):
        ball_visible = has_ball_visible(current_frame, args)
        force_fire = step_idx < args.serve_fire_steps or (
            args.auto_fire_when_ball_missing and not ball_visible
        )
        if force_fire:
            labels = (0, 0)
        else:
            labels = choose_dual_labels(args, rng, current_frame, previous_labels)
        raw_pair = labels_to_raw_actions(labels, force_fire=force_fire)

        # PettingZoo Pong's first agent controls the right screen paddle, while
        # second_0 controls the left screen paddle. The dataset stores actions
        # in screen order: [left, right].
        action_by_agent = {}
        if "first_0" in env.agents:
            action_by_agent["first_0"] = raw_pair[1]
        if "second_0" in env.agents:
            action_by_agent["second_0"] = raw_pair[0]
        if not action_by_agent:
            break

        observations, done = _step_env(env, action_by_agent)
        if AGENT_ORDER[0] not in observations:
            break

        current_frame = preprocess_frame(observations[AGENT_ORDER[0]], args.frame_size)
        dual_actions.append(labels)
        raw_actions.append(raw_pair)
        frames.append(current_frame)
        previous_labels = labels

        if done:
            break

    return Episode(
        frames=frames,
        dual_actions=dual_actions,
        raw_actions=raw_actions,
        episode_id=episode_id,
    )


def collect_sequence(
    env,
    args: Args,
    rng: np.random.Generator,
    episode_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Collect one fixed-length active gameplay sequence.

    The returned actions are exactly transition-aligned: ``dual_actions[t]`` and
    ``raw_actions[t]`` are the commands sent to move from ``frames[t]`` to
    ``frames[t + 1]``.
    """
    episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    observations = _reset_env(env, episode_seed)
    current_frame = preprocess_frame(observations[AGENT_ORDER[0]], args.frame_size)
    previous_labels = (0, 0)
    active_start_offset: int | None = None

    for step_idx in range(args.max_episode_steps):
        ball_visible = has_ball_visible(current_frame, args)
        if step_idx >= args.min_gameplay_step and (
            not args.require_ball_visible or ball_visible
        ):
            active_start_offset = step_idx
            break

        raw_pair = labels_to_raw_actions((0, 0), force_fire=True)
        action_by_agent = {}
        if "first_0" in env.agents:
            action_by_agent["first_0"] = raw_pair[1]
        if "second_0" in env.agents:
            action_by_agent["second_0"] = raw_pair[0]
        if not action_by_agent:
            return None

        observations, done = _step_env(env, action_by_agent)
        if done or AGENT_ORDER[0] not in observations:
            return None
        current_frame = preprocess_frame(observations[AGENT_ORDER[0]], args.frame_size)

    if active_start_offset is None:
        return None

    frames = [current_frame]
    dual_actions: list[tuple[int, int]] = []
    raw_actions: list[tuple[int, int]] = []

    while len(frames) < args.window_size:
        labels = choose_dual_labels(args, rng, current_frame, previous_labels)
        raw_pair = labels_to_raw_actions(labels, force_fire=False)

        action_by_agent = {}
        if "first_0" in env.agents:
            action_by_agent["first_0"] = raw_pair[1]
        if "second_0" in env.agents:
            action_by_agent["second_0"] = raw_pair[0]
        if not action_by_agent:
            return None

        observations, done = _step_env(env, action_by_agent)
        if done or AGENT_ORDER[0] not in observations:
            return None

        dual_actions.append(labels)
        raw_actions.append(raw_pair)
        current_frame = preprocess_frame(observations[AGENT_ORDER[0]], args.frame_size)
        frames.append(current_frame)
        previous_labels = labels

    frame_window = np.stack(frames, axis=0)
    dual_action_window = np.asarray(dual_actions, dtype=np.int64)
    raw_action_window = np.asarray(raw_actions, dtype=np.int64)
    visible_fraction = float(
        np.mean([has_ball_visible(frame, args) for frame in frame_window])
    )
    if args.require_ball_visible and visible_fraction < args.min_ball_visible_fraction:
        return None

    validate_arrays(frame_window, dual_action_window, raw_action_window, args)
    return (
        frame_window,
        dual_action_window,
        raw_action_window,
        episode_id,
        active_start_offset,
    )


def empty_buffers() -> SplitBuffers:
    return SplitBuffers(
        frames=[],
        dual_actions=[],
        raw_actions=[],
        episode_ids=[],
        start_offsets=[],
    )


def append_episode_windows(
    episode: Episode,
    args: Args,
    buffers: SplitBuffers,
    remaining_windows: int,
    label_histogram: np.ndarray,
) -> int:
    max_start = len(episode.frames) - args.window_size
    if max_start < 0:
        return 0

    added = 0
    first_start_offset = min(args.min_gameplay_step, max_start + 1)
    for start_offset in range(first_start_offset, max_start + 1, args.window_stride):
        if added >= remaining_windows:
            break
        end_offset = start_offset + args.window_size
        frame_window = np.stack(episode.frames[start_offset:end_offset], axis=0)
        if args.require_ball_visible:
            visible_fraction = float(
                np.mean([has_ball_visible(frame, args) for frame in frame_window])
            )
            if visible_fraction < args.min_ball_visible_fraction:
                continue
            continue
        dual_action_window = np.asarray(
            episode.dual_actions[start_offset : end_offset - 1],
            dtype=np.int64,
        )
        raw_action_window = np.asarray(
            episode.raw_actions[start_offset : end_offset - 1],
            dtype=np.int64,
        )

        validate_arrays(frame_window, dual_action_window, raw_action_window, args)
        buffers.frames.append(frame_window)
        buffers.dual_actions.append(dual_action_window)
        buffers.raw_actions.append(raw_action_window)
        buffers.episode_ids.append(episode.episode_id)
        buffers.start_offsets.append(start_offset)
        for left_label, right_label in dual_action_window:
            label_histogram[int(left_label), int(right_label)] += 1
        added += 1
    return added


def validate_arrays(
    frame_window: np.ndarray,
    dual_actions: np.ndarray,
    raw_actions: np.ndarray,
    args: Args,
) -> None:
    expected_frame_shape = (
        args.window_size,
        args.frame_size,
        args.frame_size,
        3,
    )
    expected_action_shape = (args.window_size - 1, 2)
    if frame_window.shape != expected_frame_shape:
        raise ValueError(
            f"Expected frames shape {expected_frame_shape}, got {frame_window.shape}"
        )
    if frame_window.dtype != np.uint8:
        raise ValueError(f"Expected frames dtype uint8, got {frame_window.dtype}")
    if dual_actions.shape != expected_action_shape:
        raise ValueError(
            f"Expected dual_actions shape {expected_action_shape}, got {dual_actions.shape}"
        )
    if raw_actions.shape != expected_action_shape:
        raise ValueError(
            f"Expected raw_actions shape {expected_action_shape}, got {raw_actions.shape}"
        )
    if dual_actions.dtype != np.int64:
        raise ValueError(f"Expected dual_actions dtype int64, got {dual_actions.dtype}")
    if raw_actions.dtype != np.int64:
        raise ValueError(f"Expected raw_actions dtype int64, got {raw_actions.dtype}")
    if np.any(dual_actions < 0) or np.any(dual_actions > 2):
        raise ValueError("dual_actions must be in [0, 2]")
    if np.any(raw_actions < 0) or np.any(raw_actions > 5):
        raise ValueError("raw_actions must be in [0, 5]")


def flush_buffers(
    buffers: SplitBuffers,
    split_dir: str,
    split_name: str,
    file_idx: int,
    args: Args,
    max_windows: int | None = None,
) -> dict[str, object] | None:
    if not buffers.frames:
        return None

    num_windows_to_write = len(buffers.frames)
    if max_windows is not None:
        num_windows_to_write = min(num_windows_to_write, max_windows)

    os.makedirs(split_dir, exist_ok=True)
    filename = f"chunk_{file_idx:06d}.npz"
    output_path = os.path.join(split_dir, filename)
    frames = np.stack(buffers.frames[:num_windows_to_write], axis=0).astype(np.uint8)
    dual_actions = np.stack(buffers.dual_actions[:num_windows_to_write], axis=0).astype(
        np.int64
    )
    raw_actions = np.stack(buffers.raw_actions[:num_windows_to_write], axis=0).astype(
        np.int64
    )
    episode_ids = np.asarray(
        buffers.episode_ids[:num_windows_to_write],
        dtype=np.int64,
    )
    start_offsets = np.asarray(
        buffers.start_offsets[:num_windows_to_write],
        dtype=np.int64,
    )

    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(
        output_path,
        frames=frames,
        dual_actions=dual_actions,
        raw_actions=raw_actions,
        episode_ids=episode_ids,
        start_offsets=start_offsets,
    )

    chunk_histogram = histogram_from_actions(dual_actions)
    chunk_metadata: dict[str, object] = {
        "split": split_name,
        "file": os.path.relpath(output_path, args.output_dir),
        "num_windows": int(frames.shape[0]),
        "num_transitions": int(dual_actions.shape[0] * dual_actions.shape[1]),
        "frame_shape": list(frames.shape[1:]),
        "dual_action_shape": list(dual_actions.shape[1:]),
        "label_histogram": chunk_histogram.astype(int).tolist(),
    }

    del buffers.frames[:num_windows_to_write]
    del buffers.dual_actions[:num_windows_to_write]
    del buffers.raw_actions[:num_windows_to_write]
    del buffers.episode_ids[:num_windows_to_write]
    del buffers.start_offsets[:num_windows_to_write]
    return chunk_metadata


def histogram_from_actions(dual_actions: np.ndarray) -> np.ndarray:
    histogram = np.zeros((3, 3), dtype=np.int64)
    flattened = dual_actions.reshape(-1, 2)
    for left_label, right_label in flattened:
        histogram[int(left_label), int(right_label)] += 1
    return histogram


def histogram_summary(histogram: np.ndarray) -> dict[str, object]:
    left_counts = histogram.sum(axis=1)
    right_counts = histogram.sum(axis=0)
    combined: dict[str, int] = {}
    for left_label in range(3):
        for right_label in range(3):
            key = (
                f"{NORMALIZED_ACTION_TO_NAME[left_label]}_"
                f"{NORMALIZED_ACTION_TO_NAME[right_label]}"
            )
            combined[key] = int(histogram[left_label, right_label])
    return {
        "left": {
            NORMALIZED_ACTION_TO_NAME[idx]: int(count)
            for idx, count in enumerate(left_counts)
        },
        "right": {
            NORMALIZED_ACTION_TO_NAME[idx]: int(count)
            for idx, count in enumerate(right_counts)
        },
        "combined": combined,
    }


def collect_split(
    env,
    split_name: str,
    target_windows: int,
    args: Args,
    rng: np.random.Generator,
    starting_episode_id: int,
) -> tuple[dict[str, object], int]:
    split_dir = os.path.join(args.output_dir, split_name)
    buffers = empty_buffers()
    chunks: list[dict[str, object]] = []
    label_histogram = np.zeros((3, 3), dtype=np.int64)
    file_idx = 0
    num_windows = 0
    episode_id = starting_episode_id

    progress = tqdm(total=target_windows, desc=f"Collecting {split_name}", unit="window")
    while num_windows < target_windows:
        if episode_id - starting_episode_id >= args.max_episodes_per_split:
            raise RuntimeError(
                f"Could only collect {num_windows}/{target_windows} windows for "
                f"{split_name} after {args.max_episodes_per_split} episodes. "
                "Relax gameplay filters or reduce sequence length."
            )
        sequence = collect_sequence(env, args, rng, episode_id)
        episode_id += 1
        if sequence is None:
            continue

        frame_window, dual_action_window, raw_action_window, source_episode_id, start_offset = sequence
        buffers.frames.append(frame_window)
        buffers.dual_actions.append(dual_action_window)
        buffers.raw_actions.append(raw_action_window)
        buffers.episode_ids.append(source_episode_id)
        buffers.start_offsets.append(start_offset)
        for left_label, right_label in dual_action_window:
            label_histogram[int(left_label), int(right_label)] += 1
        num_windows += 1
        progress.update(1)

        while len(buffers.frames) >= args.windows_per_file:
            chunk_metadata = flush_buffers(
                buffers,
                split_dir,
                split_name,
                file_idx,
                args,
                max_windows=args.windows_per_file,
            )
            if chunk_metadata is not None:
                chunks.append(chunk_metadata)
                file_idx += 1
    progress.close()

    chunk_metadata = flush_buffers(buffers, split_dir, split_name, file_idx, args)
    if chunk_metadata is not None:
        chunks.append(chunk_metadata)

    split_metadata: dict[str, object] = {
        "num_windows": int(num_windows),
        "num_transitions": int(num_windows * (args.window_size - 1)),
        "num_files": len(chunks),
        "label_histogram": label_histogram.astype(int).tolist(),
        "label_histogram_summary": histogram_summary(label_histogram),
        "chunks": chunks,
    }
    return split_metadata, episode_id


def save_preview(output_dir: str, args: Args) -> str | None:
    train_dir = os.path.join(output_dir, "train")
    if not os.path.isdir(train_dir) or args.preview_samples <= 0:
        return None

    chunk_files = sorted(
        filename for filename in os.listdir(train_dir) if filename.endswith(".npz")
    )
    if not chunk_files:
        return None

    first_chunk_path = os.path.join(train_dir, chunk_files[0])
    with np.load(first_chunk_path) as shard:
        frames = shard["frames"][: args.preview_samples]
        dual_actions = shard["dual_actions"][: args.preview_samples]

    if frames.size == 0:
        return None

    rows = frames.shape[0]
    columns = frames.shape[1]
    cell_size = args.frame_size
    label_band = 18
    preview = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_band)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(preview)

    for row_idx in range(rows):
        row_top = row_idx * (cell_size + label_band)
        for frame_idx in range(columns):
            image = Image.fromarray(frames[row_idx, frame_idx])
            preview.paste(image, (frame_idx * cell_size, row_top + label_band))
            if frame_idx < columns - 1:
                left_label, right_label = dual_actions[row_idx, frame_idx]
                text = (
                    f"L:{NORMALIZED_ACTION_TO_NAME[int(left_label)][0]} "
                    f"R:{NORMALIZED_ACTION_TO_NAME[int(right_label)][0]}"
                )
                draw.text(
                    (frame_idx * cell_size + 2, row_top + 2),
                    text,
                    fill=(255, 255, 255),
                )

    preview_dir = os.path.join(output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, "train_preview.png")
    preview.save(preview_path)
    return preview_path


def build_metadata(
    args: Args,
    split_metadata: dict[str, dict[str, object]],
    preview_path: str | None,
) -> dict[str, object]:
    return {
        "dataset_name": "two_player_pong_dual_controls",
        "format_version": 1,
        "frame_layout": "NTHWC",
        "frame_spacing": 1,
        "fps": 60,
        "env_step_semantics": "one PettingZoo parallel_env.step call, one ALE act call, no frame skip",
        "frame_dtype": "uint8",
        "frame_shape": [args.window_size, args.frame_size, args.frame_size, 3],
        "dual_action_shape": [args.window_size - 1, 2],
        "dual_action_dtype": "int64",
        "normalized_action_mapping": {
            str(key): value for key, value in NORMALIZED_ACTION_TO_NAME.items()
        },
        "raw_action_mapping": {
            str(key): value for key, value in RAW_ACTION_TO_NAME.items()
        },
        "normalized_to_raw_action": {
            str(key): value for key, value in NORMALIZED_ACTION_TO_RAW.items()
        },
        "agent_order": list(AGENT_ORDER),
        "paddle_order": list(PADDLE_ORDER),
        "agent_to_paddle": AGENT_TO_PADDLE,
        "collection": asdict(args),
        "splits": split_metadata,
        "preview_path": preview_path,
    }


def write_metadata(output_dir: str, metadata: dict[str, object]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
    return metadata_path


def validate_args(args: Args) -> None:
    if args.window_size < 2:
        raise ValueError("window_size must be at least 2")
    if args.window_stride < 1:
        raise ValueError("window_stride must be at least 1")
    if args.windows_per_file < 1:
        raise ValueError("windows_per_file must be at least 1")
    if args.frame_size < 1:
        raise ValueError("frame_size must be positive")
    if args.min_gameplay_step < 0:
        raise ValueError("min_gameplay_step must be nonnegative")
    if args.max_episodes_per_split < 1:
        raise ValueError("max_episodes_per_split must be at least 1")
    if args.ball_threshold < 0 or args.ball_threshold > 255:
        raise ValueError("ball_threshold must be in [0, 255]")
    if not 0.0 <= args.min_ball_visible_fraction <= 1.0:
        raise ValueError("min_ball_visible_fraction must be in [0, 1]")
    if args.ball_min_pixels < 1:
        raise ValueError("ball_min_pixels must be at least 1")
    if args.ball_max_component_pixels < args.ball_min_pixels:
        raise ValueError("ball_max_component_pixels must be >= ball_min_pixels")
    if (
        args.num_windows_train < 0
        or args.num_windows_val < 0
        or args.num_windows_test < 0
    ):
        raise ValueError("num_windows_* values must be nonnegative")
    if not 0.0 <= args.sticky_action_prob <= 1.0:
        raise ValueError("sticky_action_prob must be in [0, 1]")


def prepare_output_dir(args: Args) -> None:
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    validate_args(args)
    prepare_output_dir(args)

    rng = np.random.default_rng(args.seed)
    env = pong_v3.parallel_env(num_players=2, render_mode=None)
    split_targets = {
        "train": args.num_windows_train,
        "val": args.num_windows_val,
        "test": args.num_windows_test,
    }
    split_metadata: dict[str, dict[str, object]] = {}
    episode_id = 0

    try:
        for split_name, target_windows in split_targets.items():
            if target_windows <= 0:
                split_metadata[split_name] = {
                    "num_windows": 0,
                    "num_transitions": 0,
                    "num_files": 0,
                    "label_histogram": np.zeros((3, 3), dtype=np.int64).tolist(),
                    "label_histogram_summary": histogram_summary(
                        np.zeros((3, 3), dtype=np.int64)
                    ),
                    "chunks": [],
                }
                continue
            metadata, episode_id = collect_split(
                env,
                split_name,
                target_windows,
                args,
                rng,
                episode_id,
            )
            split_metadata[split_name] = metadata
    finally:
        env.close()

    preview_path = save_preview(args.output_dir, args)
    metadata = build_metadata(args, split_metadata, preview_path)
    metadata_path = write_metadata(args.output_dir, metadata)

    logger.info("Wrote metadata to %s", metadata_path)
    if preview_path is not None:
        logger.info("Wrote preview to %s", preview_path)
    for split_name, metadata_for_split in split_metadata.items():
        logger.info(
            "%s: %s windows, %s transitions, %s files",
            split_name,
            metadata_for_split["num_windows"],
            metadata_for_split["num_transitions"],
            metadata_for_split["num_files"],
        )
        logger.info(
            "%s label histogram: %s",
            split_name,
            metadata_for_split["label_histogram_summary"],
        )


if __name__ == "__main__":
    main()
