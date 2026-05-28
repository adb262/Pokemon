#!/usr/bin/env python3
"""Interactive Pong action-mapping + dynamics inference.

Controls:
    w/s: left paddle up/down
    i/k: right paddle up/down
    q or Esc: quit
    r: reset to the seed clip
"""

import logging
import os
import select
import sys
import termios
import json
import threading
import time
import tty
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
import tyro

from action_mapping.model import ActionMappingModel
from dynamics_model.checkpoints import adapt_state_dict_to_model
from dynamics_model.model import DynamicsModel
from latent_action_model.model import LatentActionVQVAE
from scripts.video_tokenizer.post_train_tokenizer import (
    PostTrainTokenizerConfig,
    _apply_dynamics_checkpoint_config,
    _load_checkpoint_config,
    load_frozen_dynamics_stack,
)
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class InteractiveInferenceConfig:
    dynamics_model_checkpoint_path: str = (
        "dynamics_model_pong_w_tokenizer_v2_256_scheduled_opt_longer_eval_128_d_action_16_frames_1_denoising_step_512_dynamics_anchor_action_fixed/checkpoint_latest.pt"
    )
    action_model_checkpoint_path: str = (
        "dynamics_model_pong_w_tokenizer_v2_256_scheduled_opt_longer_eval_128_d_action_16_frames_1_denoising_step_512_dynamics_anchor_action_fixed/action_model/checkpoint_latest.pt"
    )
    tokenizer_checkpoint_path: str = "post_train_tokenizer_500k/checkpoint_epoch0_batch16003.pt"
    # Optional override for the tokenizer used by the *dynamics model* when
    # encoding its rolling context. The dynamics model was trained against
    # a specific tokenizer; if the action mapping model was trained against
    # the same one (the gold setup), leave this `None` to reuse
    # `tokenizer_checkpoint_path`. Set this to point at a different
    # tokenizer checkpoint if your dynamics+action-mapping training stack
    # used a tokenizer that differs from the post-trained rollout decoder.
    dynamics_tokenizer_checkpoint_path: str | None = None
    action_mapping_checkpoint_path: str = "action_mapping_1_layer/checkpoint_best.pt"

    seed_data_dir: str = "data/two_player_pong"
    seed_split: Literal["train", "val", "test"] = "train"
    seed_window_index: int = 0
    # 0 means "auto": seed with enough real frames to fill the trained
    # action-mapping and dynamics context. Set to 1 for cold-start stress tests.
    rollout_seed_frames: int = 0
    image_size: int | None = None
    denoising_steps: int = 1
    dynamics_context_frames: int = 16
    use_bf16: bool = True
    input_interval_ms: float = 200.0
    display_scale: int = 4
    display_backend: Literal["web", "opencv"] = "web"
    web_host: str = "127.0.0.1"
    web_port: int = 8766
    timing_log_interval: int = 30
    seed: int = 42
    interactive_device: str = (
        "cuda:1"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class RawKeyboard:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: Any | None = None

    def __enter__(self) -> "RawKeyboard":
        if not sys.stdin.isatty():
            return self
        fd = sys.stdin.fileno()
        self._fd = fd
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_available(self) -> str:
        if not sys.stdin.isatty():
            return ""
        chars: list[str] = []
        while True:
            readable, _writable, _error = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            chars.append(sys.stdin.read(1))
        return "".join(chars)


WEB_PAGE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Action Mapping Pong</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    #wrap { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 16px; }
    #wrap:focus { outline: 2px solid #9ee; outline-offset: -6px; }
    img { image-rendering: pixelated; border: 1px solid #555; background: #000; }
    code { color: #9ee; }
  </style>
</head>
<body>
  <div id="wrap" tabindex="0">
    <img id="frame" src="/frame.jpg" alt="generated frame" />
    <div>
      Controls: <code>w/s</code> left up/down, <code>i/k</code> right up/down,
      <code>r</code> reset, <code>q</code> quit
    </div>
    <div id="status">Click the viewer, then use controls...</div>
  </div>
  <script>
    const keys = new Set();
    const wrap = document.getElementById("wrap");
    const frame = document.getElementById("frame");
    const status = document.getElementById("status");

    function currentAction() {
      let left = 0;
      let right = 0;
      if (keys.has("w")) left = 1;
      if (keys.has("s")) left = 2;
      if (keys.has("i")) right = 1;
      if (keys.has("k")) right = 2;
      return left * 3 + right;
    }

    async function postJson(path, payload) {
      try {
        await fetch(path, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
          keepalive: true,
        });
      } catch (_err) {
        status.textContent = "Connection lost";
      }
    }

    function sendCurrentAction() {
      const action = currentAction();
      postJson("/input", {action});
      status.textContent = "action=" + action + " updated=" + new Date().toLocaleTimeString();
    }

    function refreshFrame() {
      frame.src = "/frame.jpg?ts=" + Date.now();
    }

    window.addEventListener("load", () => {
      wrap.focus();
      sendCurrentAction();
      refreshFrame();
    });

    document.addEventListener("click", () => {
      wrap.focus();
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        keys.clear();
        sendCurrentAction();
      }
    });

    document.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      if ("wsik".includes(key)) {
        keys.add(key);
        event.preventDefault();
        sendCurrentAction();
      } else if (key === "r") {
        event.preventDefault();
        postJson("/reset", {});
      } else if (key === "q" || event.key === "Escape") {
        if (event.repeat) return;
        event.preventDefault();
        postJson("/quit", {});
      }
    });

    document.addEventListener("keyup", (event) => {
      const key = event.key.toLowerCase();
      if ("wsik".includes(key)) {
        keys.delete(key);
        event.preventDefault();
        sendCurrentAction();
      }
    });

    setInterval(() => {
      sendCurrentAction();
    }, 30);

    setInterval(() => {
      refreshFrame();
    }, 100);
  </script>
</body>
</html>
"""


class WebDisplayState:
    def __init__(self, input_timeout_seconds: float) -> None:
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.latest_jpeg: bytes | None = None
        self.joint_action = 0
        self.pending_joint_action = 0
        self.last_logged_action = 0
        self.input_post_count = 0
        self.frame_publish_count = 0
        self.frame_get_count = 0
        self.stream_client_count = 0
        self.stream_frame_count = 0
        self.last_input_time = time.monotonic()
        self.reset_requested = False
        self.quit_requested = False
        self.input_timeout_seconds = input_timeout_seconds

    def update_frame(self, display: np.ndarray) -> None:
        success, encoded = cv2.imencode(".jpg", display)
        if not success:
            return
        with self.lock:
            self.latest_jpeg = encoded.tobytes()
            self.frame_publish_count += 1
            self.frame_condition.notify_all()

    def wait_for_frame(
        self,
        last_seen_count: int,
        timeout_seconds: float,
    ) -> tuple[bytes | None, int]:
        with self.frame_condition:
            if self.frame_publish_count <= last_seen_count:
                self.frame_condition.wait(timeout=timeout_seconds)
            return self.latest_jpeg, self.frame_publish_count

    def record_frame_get(self) -> None:
        with self.lock:
            self.frame_get_count += 1

    def record_stream_client(self) -> None:
        with self.lock:
            self.stream_client_count += 1
            logger.info("MJPEG stream client connected (clients=%s)", self.stream_client_count)

    def record_stream_frame(self) -> None:
        with self.lock:
            self.stream_frame_count += 1

    def set_action(self, joint_action: int) -> None:
        clamped_action = max(0, min(8, int(joint_action)))
        with self.lock:
            self.joint_action = clamped_action
            if clamped_action != 0:
                self.pending_joint_action = clamped_action
            self.input_post_count += 1
            self.last_input_time = time.monotonic()
            if clamped_action != self.last_logged_action:
                logger.info("Web input action changed: %s -> %s", self.last_logged_action, clamped_action)
                self.last_logged_action = clamped_action

    def request_reset(self) -> None:
        with self.lock:
            if self.reset_requested:
                return
            self.reset_requested = True
            logger.info("Web reset requested")

    def request_quit(self) -> None:
        with self.lock:
            self.quit_requested = True
            logger.info("Web quit requested")

    def consume_controls(self) -> tuple[int, bool, bool, int, int, int, int, int]:
        with self.lock:
            action = self.joint_action
            if time.monotonic() - self.last_input_time > self.input_timeout_seconds:
                action = 0
            if action == 0 and self.pending_joint_action != 0:
                action = self.pending_joint_action
            self.pending_joint_action = 0
            reset = self.reset_requested
            quit_requested = self.quit_requested
            input_post_count = self.input_post_count
            frame_publish_count = self.frame_publish_count
            frame_get_count = self.frame_get_count
            stream_client_count = self.stream_client_count
            stream_frame_count = self.stream_frame_count
            self.reset_requested = False
            return (
                action,
                reset,
                quit_requested,
                input_post_count,
                frame_publish_count,
                frame_get_count,
                stream_client_count,
                stream_frame_count,
            )


class WebDisplayHTTPServer(ThreadingHTTPServer):
    state: WebDisplayState

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        state: WebDisplayState,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.state = state


class WebDisplayHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/stream.mjpg"):
            self._serve_stream()
            return
        if self.path.startswith("/frame.jpg"):
            self._serve_frame()
            return
        self._serve_html()

    def do_POST(self) -> None:
        state = cast(WebDisplayHTTPServer, self.server).state
        if self.path == "/input":
            content_length = int(self.headers.get("Content-Length", "0"))
            payload_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(payload_bytes.decode("utf-8"))
                state.set_action(int(payload.get("action", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                state.set_action(0)
            self._send_empty_response()
            return
        if self.path == "/reset":
            state.request_reset()
            self._send_empty_response()
            return
        if self.path == "/quit":
            state.request_quit()
            self._send_empty_response()
            return
        self.send_error(404)

    def _serve_html(self) -> None:
        content = WEB_PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_frame(self) -> None:
        state = cast(WebDisplayHTTPServer, self.server).state
        state.record_frame_get()
        with state.lock:
            jpeg = state.latest_jpeg
        if jpeg is None:
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_stream(self) -> None:
        state = cast(WebDisplayHTTPServer, self.server).state
        state.record_stream_client()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_seen_count = -1
        while not state.quit_requested:
            jpeg, frame_count = state.wait_for_frame(
                last_seen_count,
                timeout_seconds=1.0,
            )
            if jpeg is None or frame_count == last_seen_count:
                continue
            last_seen_count = frame_count
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                state.record_stream_frame()
            except (BrokenPipeError, ConnectionResetError):
                logger.info("MJPEG stream client disconnected")
                break

    def _send_empty_response(self) -> None:
        self.send_response(204)
        self.end_headers()


def start_web_display(
    host: str,
    port: int,
    input_interval_ms: float,
) -> tuple[WebDisplayHTTPServer, WebDisplayState]:
    timeout = max(input_interval_ms, 1.0) / 1000.0 * 3.0
    state = WebDisplayState(input_timeout_seconds=timeout)
    server = WebDisplayHTTPServer((host, port), WebDisplayHandler, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Open the interactive viewer at http://%s:%s", host, server.server_port)
    return server, state


def build_post_train_rollout_config(
    config: InteractiveInferenceConfig,
    device: torch.device,
) -> PostTrainTokenizerConfig:
    post_train_config = PostTrainTokenizerConfig(
        tokenizer_checkpoint_path=config.tokenizer_checkpoint_path,
        dynamics_tokenizer_checkpoint_path=config.dynamics_tokenizer_checkpoint_path,
        dynamics_model_checkpoint_path=config.dynamics_model_checkpoint_path,
        action_model_checkpoint_path=config.action_model_checkpoint_path,
        post_train_frames=max(
            config.rollout_seed_frames + 1,
            config.dynamics_context_frames,
        ),
        dynamics_context_frames=config.dynamics_context_frames,
        rollout_seed_frames=config.rollout_seed_frames,
        max_denoising_steps=config.denoising_steps,
        seed=config.seed,
        device=str(device),
        use_bf16=config.use_bf16,
        use_compile=False,
        logging_backend="none",
        use_wandb=False,
    )
    dynamics_config = _load_checkpoint_config(config.dynamics_model_checkpoint_path)
    action_config = _load_checkpoint_config(config.action_model_checkpoint_path)
    _apply_dynamics_checkpoint_config(
        post_train_config,
        dynamics_config,
        action_config,
    )
    if config.image_size is not None:
        post_train_config.image_size = config.image_size
    post_train_config.dynamics_context_frames = min(
        config.dynamics_context_frames,
        post_train_config.num_images_in_video,
    )
    return post_train_config


def load_frozen_pipeline_models(
    config: InteractiveInferenceConfig,
    device: torch.device,
) -> tuple[VideoTokenizer, LatentActionVQVAE, DynamicsModel]:
    post_train_config = build_post_train_rollout_config(config, device)
    rollout_tokenizer, tokenizer_config = load_model_from_checkpoint(
        config.tokenizer_checkpoint_path,
        device,
    )
    if config.image_size is not None:
        post_train_config.image_size = config.image_size
    else:
        post_train_config.image_size = tokenizer_config.image_size
        post_train_config.patch_size = tokenizer_config.patch_size
        post_train_config.bins = list(tokenizer_config.bins)

    dynamics_model = load_frozen_dynamics_stack(post_train_config, device)
    action_model = dynamics_model.action_model

    for module in (rollout_tokenizer, action_model, dynamics_model):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    return rollout_tokenizer, action_model, dynamics_model


def load_action_mapping_model(
    checkpoint_path: str,
    action_vocab_size: int,
    dynamics_d_model: int,
    dynamics_num_images_in_video: int,
    device: torch.device,
) -> ActionMappingModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    # `scripts/action_mapping/train.py` lets `d_model` default to `None` at
    # config time and resolves it to `dynamics_model.d_model` at runtime
    # without writing the resolved value back into the saved config. So
    # `d_model is None` here means "use the dynamics d_model", matching
    # training behavior. The remaining architectural keys must be present
    # or we would silently fall back to training defaults that almost
    # certainly do not match the checkpoint (e.g. max_sequence_length=5 vs
    # a trained 16).
    required_keys = (
        "num_input_actions",
        "max_sequence_length",
        "num_heads",
        "num_layers",
    )
    missing_keys = [key for key in required_keys if key not in checkpoint_config]
    if missing_keys:
        raise ValueError(
            "Action mapping checkpoint is missing required config keys "
            f"{missing_keys}. Re-export the checkpoint with its training "
            "config so the loader does not silently fall back to defaults."
        )

    checkpoint_d_model = checkpoint_config.get("d_model")
    d_model = (
        dynamics_d_model
        if checkpoint_d_model is None
        else int(checkpoint_d_model)
    )
    num_input_actions = int(checkpoint_config["num_input_actions"])
    max_sequence_length = int(checkpoint_config["max_sequence_length"])
    num_heads = int(checkpoint_config["num_heads"])
    num_layers = int(checkpoint_config["num_layers"])

    # `scripts/action_mapping/train.py` enforces this equality at training
    # time (see the assertion in `main`), so any mismatch here means the
    # checkpoint and dynamics model were not trained against each other.
    if max_sequence_length != dynamics_num_images_in_video:
        raise ValueError(
            "Action mapping max_sequence_length must match dynamics "
            f"num_images_in_video: got {max_sequence_length} vs "
            f"{dynamics_num_images_in_video}. Was this action mapping "
            "checkpoint trained against a different dynamics model?"
        )
    if d_model != dynamics_d_model:
        raise ValueError(
            "Action mapping d_model must match dynamics d_model: "
            f"got {d_model} vs {dynamics_d_model}"
        )

    logger.info(
        "Loaded action mapping checkpoint from %s: d_model=%s "
        "num_input_actions=%s max_sequence_length=%s num_heads=%s "
        "num_layers=%s action_vocab_size=%s",
        checkpoint_path,
        d_model,
        num_input_actions,
        max_sequence_length,
        num_heads,
        num_layers,
        action_vocab_size,
    )

    model = ActionMappingModel(
        num_input_actions=num_input_actions,
        num_output_actions=action_vocab_size,
        max_sequence_length=max_sequence_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)
    state_dict = adapt_state_dict_to_model(checkpoint["model_state_dict"], model)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def count_seed_windows(config: InteractiveInferenceConfig) -> int:
    split_dir = Path(config.seed_data_dir) / config.seed_split
    shard_paths = sorted(split_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No seed .npz files found in {split_dir}")

    total_windows = 0
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            total_windows += int(shard["frames"].shape[0])
    return total_windows


def sample_next_seed_window_index(
    *,
    rng: np.random.Generator,
    total_windows: int,
    current_index: int,
) -> int:
    if total_windows < 1:
        raise ValueError("Cannot sample a seed window from an empty dataset")
    if total_windows == 1:
        return 0

    sampled_index = int(rng.integers(0, total_windows - 1))
    if sampled_index >= current_index:
        sampled_index += 1
    return sampled_index


def load_seed_video(
    config: InteractiveInferenceConfig,
    num_frames: int,
    image_size: int,
    device: torch.device,
    seed_window_index: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    split_dir = Path(config.seed_data_dir) / config.seed_split
    shard_paths = sorted(split_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No seed .npz files found in {split_dir}")

    requested_index = (
        config.seed_window_index if seed_window_index is None else seed_window_index
    )
    remaining_index = requested_index
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            frames = shard["frames"]
            if remaining_index >= frames.shape[0]:
                remaining_index -= frames.shape[0]
                continue
            seed_frames = frames[remaining_index, :num_frames]
            seed_dual_actions = shard["dual_actions"][remaining_index, : num_frames - 1]
            break
    else:
        raise IndexError(
            f"seed_window_index={requested_index} is out of range for {split_dir}"
        )

    if seed_frames.shape[0] < num_frames:
        raise ValueError(
            f"Seed clip only has {seed_frames.shape[0]} frames, need {num_frames}"
        )
    if seed_dual_actions.shape[0] < num_frames - 1:
        raise ValueError(
            f"Seed clip only has {seed_dual_actions.shape[0]} actions, "
            f"need {num_frames - 1}"
        )

    video = torch.from_numpy(seed_frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    if video.shape[-2:] != (image_size, image_size):
        video = torch.nn.functional.interpolate(
            video,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    joint_actions = seed_dual_actions[:, 0] * 3 + seed_dual_actions[:, 1]
    source_action_history = [int(action) for action in joint_actions.tolist()]
    return video.unsqueeze(0).to(device), source_action_history


def read_joint_action(chars: str) -> tuple[int, bool, bool]:
    left_action = 0
    right_action = 0
    reset = False
    quit_requested = False

    for char in chars.lower():
        if char in ("q", "\x1b"):
            quit_requested = True
        elif char == "r":
            reset = True
        elif char == "w":
            left_action = 1
        elif char == "s":
            left_action = 2
        elif char == "i":
            right_action = 1
        elif char == "k":
            right_action = 2

    return left_action * 3 + right_action, reset, quit_requested


@torch.no_grad()
def compute_video_token_latents(
    dynamics_model: DynamicsModel,
    video_context: torch.Tensor,
) -> torch.Tensor:
    quantized = dynamics_model.tokenizer.encode(video_context)
    codes = dynamics_model.tokenizer.quantized_value_to_codes(quantized).long()
    return dynamics_model.tokenizer_embedding(codes)


def score_candidate_frame_sequence_fit(
    generated_video: torch.Tensor,
    candidate_frames: torch.Tensor,
) -> torch.Tensor:
    """Score candidate next frames by cheap temporal consistency.

    Higher is better. The score is a 0-100-ish transform of pixel-space
    temporal error, so good candidates do not disappear as rounded 0.00 values.
    It favors candidates whose motion continues the recent generated sequence,
    with extra weight around changing pixels so the static Pong background does
    not dominate.
    """
    if generated_video.shape[0] != 1:
        raise ValueError(
            "Interactive candidate scoring expects a single generated video, "
            f"got batch size {generated_video.shape[0]}"
        )
    if candidate_frames.dim() != 4:
        raise ValueError(
            "candidate_frames must have shape (num_candidates, C, H, W), "
            f"got {tuple(candidate_frames.shape)}"
        )

    history = generated_video[0]
    last_frame = history[-1].unsqueeze(0)
    candidate_delta = candidate_frames - last_frame
    if history.shape[0] < 2:
        pixel_space_loss = candidate_delta.square().flatten(1).mean(dim=1) * 255.0**2
        return 100.0 / (1.0 + pixel_space_loss)

    previous_delta = (history[-1] - history[-2]).unsqueeze(0)
    expected_next_frame = (last_frame + previous_delta).clamp(0.0, 1.0)

    previous_motion = previous_delta.abs().mean(dim=1, keepdim=True)
    candidate_motion = candidate_delta.abs().mean(dim=1, keepdim=True)
    motion_weights = 1.0 + 8.0 * (previous_motion + candidate_motion)

    sequence_loss = (
        (candidate_frames - expected_next_frame).square() * motion_weights
    ).flatten(1).mean(dim=1)
    if history.shape[0] >= 3:
        previous_previous_delta = (history[-2] - history[-3]).unsqueeze(0)
        previous_acceleration = previous_delta - previous_previous_delta
        candidate_acceleration = candidate_delta - previous_delta
        acceleration_loss = (
            (candidate_acceleration - previous_acceleration).square() * motion_weights
        ).flatten(1).mean(dim=1)
        sequence_loss = sequence_loss + 0.25 * acceleration_loss

    pixel_space_loss = sequence_loss * 255.0**2
    return 100.0 / (1.0 + pixel_space_loss)


@torch.no_grad()
def predict_next_interactive_frame(
    *,
    action_mapping_model: torch.nn.Module,
    dynamics_model: DynamicsModel,
    rollout_tokenizer: VideoTokenizer,
    generated_video: torch.Tensor,
    source_action_history: list[int],
    latent_action_history: list[int],
    mapping_sequence_length: int,
    denoising_steps: int,
    use_amp: bool,
) -> tuple[torch.Tensor, int, float, list[int]]:
    # Construct a fresh `torch.autocast` for every `with` site rather than
    # reusing a single instance. Reusing the same context-manager instance
    # across multiple `with` blocks works only when no other context
    # manager mutates the global autocast state between them. Use separate
    # contexts for the action mapper and dynamics step so both can follow the
    # configured precision without sharing cached casts.
    #
    # ``cache_enabled=False`` disables PyTorch's per-tensor cached_cast
    # weight cache. The cache is process-wide and is observed to leave
    # entries pointing at deallocated bf16 weights once the second GPU
    # worker thread starts entering/exiting its own autocast blocks; the
    # next eager `linear` on the main thread then hits a "weight is float,
    # input is bfloat16" mismatch on the cached entry. Re-casting weights
    # every call is a few percent slower but eliminates the cross-thread
    # cache hazard.
    def make_amp_ctx() -> torch.autocast:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp and generated_video.device.type == "cuda",
            cache_enabled=False,
        )

    # Training feeds the action mapper the first T-1 frame latents and T-1
    # source actions from each T-frame window, then supervises the T-1 latent
    # transition tokens. Mirror that here: the newly-read source action is
    # paired with the latest generated frame to predict the next latent action.
    mapping_context_length = min(
        generated_video.shape[1],
        max(mapping_sequence_length - 1, 1),
    )
    mapping_context_frames = generated_video[:, -mapping_context_length:]
    recent_actions = source_action_history[-mapping_context_length:]
    if len(recent_actions) != mapping_context_frames.shape[1]:
        raise ValueError(
            "Source action history must match the generated frame count "
            "at each tick: "
            f"frames={mapping_context_frames.shape[1]}, "
            f"actions={len(recent_actions)}"
        )
    source_actions = torch.tensor(
        recent_actions,
        dtype=torch.long,
        device=generated_video.device,
    ).unsqueeze(0)

    # `compute_video_token_latents` stays outside autocast to mirror
    # `train.py` (it runs the dynamics tokenizer under `torch.no_grad`
    # without amp); the action mapping forward must run under amp so the
    # bf16 numerics match training.
    video_token_latents = compute_video_token_latents(
        dynamics_model,
        mapping_context_frames,
    )
    with make_amp_ctx():
        logits = action_mapping_model(video_token_latents, source_actions)

    if logits.shape[0] != 1:
        raise ValueError(
            "Interactive inference expects batch size 1 from action mapping, "
            f"got {logits.shape[0]}"
        )
    mapped_action_tensor = logits[:, -1].argmax(dim=-1).long()

    expected_latent_actions = generated_video.shape[1] - 1
    if len(latent_action_history) != expected_latent_actions:
        raise ValueError(
            "Latent action history must contain one action per generated "
            "transition before the next tick: "
            f"frames={generated_video.shape[1]}, "
            f"latent_actions={len(latent_action_history)}"
        )
    dynamics_context_length = min(
        generated_video.shape[1],
        dynamics_model.num_images_in_video - 1,
    )
    num_context_actions = max(dynamics_context_length - 1, 0)
    if num_context_actions > 0:
        context_actions = torch.tensor(
            latent_action_history[-num_context_actions:],
            dtype=torch.long,
            device=generated_video.device,
        ).unsqueeze(0)
    else:
        context_actions = torch.empty(
            1, 0, dtype=torch.long, device=generated_video.device
        )
    with make_amp_ctx():
        next_frame = dynamics_model.predict_next_frame(
            generated_video,
            mapped_action_tensor,
            max_steps=denoising_steps,
            decode_tokenizer=rollout_tokenizer,
            context_actions=context_actions,
        )

    quality_scores = score_candidate_frame_sequence_fit(
        generated_video=generated_video,
        candidate_frames=next_frame,
    )
    mapped_action = int(mapped_action_tensor[0].item())
    selected_quality = float(quality_scores[0].item())
    candidate_action_list = [mapped_action]

    return next_frame, mapped_action, selected_quality, candidate_action_list


def frame_to_display(frame: torch.Tensor, scale: int) -> np.ndarray:
    image = (
        frame.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if scale != 1:
        image = cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return image


def trim_rolling_state(
    *,
    generated_video: torch.Tensor,
    source_action_history: list[int],
    latent_action_history: list[int],
    rolling_window_length: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    return (
        generated_video[:, -rolling_window_length:],
        source_action_history[-(rolling_window_length - 1) :],
        latent_action_history[-(rolling_window_length - 1) :],
    )


@dataclass
class InteractiveRolloutState:
    generated_video: torch.Tensor
    source_action_history: list[int]
    latent_action_history: list[int]
    latest_interactive_display: np.ndarray


@torch.no_grad()
def initialize_interactive_rollout_state(
    *,
    config: InteractiveInferenceConfig,
    action_model: LatentActionVQVAE,
    image_size: int,
    interactive_device: torch.device,
    seed_window_index: int,
) -> InteractiveRolloutState:
    if config.rollout_seed_frames < 1:
        raise ValueError(
            f"rollout_seed_frames must be at least 1, got {config.rollout_seed_frames}"
        )

    seed_video, seed_source_actions = load_seed_video(
        config,
        config.rollout_seed_frames,
        image_size,
        interactive_device,
        seed_window_index=seed_window_index,
    )
    generated_video = seed_video.clone()
    source_action_history = list(seed_source_actions)
    if generated_video.shape[1] > 1:
        seed_action_encoded = action_model.encode(generated_video)
        seed_latent_actions = action_model.get_action_sequence(
            seed_action_encoded
        ).long()
        latent_action_history = [
            int(action) for action in seed_latent_actions[0].tolist()
        ]
    else:
        latent_action_history = []
    latest_interactive_display = frame_to_display(
        generated_video[0, -1],
        config.display_scale,
    )

    return InteractiveRolloutState(
        generated_video=generated_video,
        source_action_history=source_action_history,
        latent_action_history=latent_action_history,
        latest_interactive_display=latest_interactive_display,
    )


def main(config: InteractiveInferenceConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    interactive_device = torch.device(config.interactive_device)
    logger.info("Using device: interactive=%s", interactive_device)
    if not os.path.exists(config.action_mapping_checkpoint_path):
        logger.warning(
            "Action mapping checkpoint does not exist yet: %s",
            config.action_mapping_checkpoint_path,
        )

    tokenizer, action_model, dynamics_model = load_frozen_pipeline_models(
        config,
        interactive_device,
    )
    action_mapping_model = load_action_mapping_model(
        config.action_mapping_checkpoint_path,
        action_model.action_vocab_size,
        dynamics_model.d_model,
        dynamics_model.num_images_in_video,
        interactive_device,
    )
    mapping_sequence_length = action_mapping_model.max_sequence_length

    interactive_dynamics_context_frames = min(
        config.dynamics_context_frames,
        dynamics_model.num_images_in_video,
    )
    interactive_use_amp = config.use_bf16 and interactive_device.type == "cuda"
    image_size = (
        config.image_size if config.image_size is not None else action_model.image_height
    )
    full_context_seed_frames = max(
        max(mapping_sequence_length - 1, 1),
        max(dynamics_model.num_images_in_video - 1, 1),
        max(tokenizer.decoder.num_images_in_video - 1, 1),
    )
    if config.rollout_seed_frames <= 0:
        config.rollout_seed_frames = full_context_seed_frames
    elif config.rollout_seed_frames < full_context_seed_frames:
        logger.warning(
            "rollout_seed_frames=%s is shorter than the trained full context "
            "(%s frames); expect early-tick quality degradation.",
            config.rollout_seed_frames,
            full_context_seed_frames,
        )
    rolling_window_length = max(
        interactive_dynamics_context_frames,
        mapping_sequence_length,
        tokenizer.decoder.num_images_in_video,
    )
    seed_window_count = count_seed_windows(config)
    seed_rng = np.random.default_rng(config.seed)
    current_seed_window_index = config.seed_window_index
    rollout_state = initialize_interactive_rollout_state(
        config=config,
        action_model=action_model,
        image_size=image_size,
        interactive_device=interactive_device,
        seed_window_index=current_seed_window_index,
    )
    generated_video = rollout_state.generated_video
    source_action_history = rollout_state.source_action_history
    latent_action_history = rollout_state.latent_action_history
    latest_interactive_display = rollout_state.latest_interactive_display
    logger.info(
        "Initialized rolling windows: frames=%s source_actions=%s latent_actions=%s "
        "use_bf16=%s",
        tuple(generated_video.shape),
        len(source_action_history),
        len(latent_action_history),
        interactive_use_amp,
    )

    window_name = "Action Mapping Pong - w/s left, i/k right, r reset, q quit"
    web_server: WebDisplayHTTPServer | None = None
    web_state: WebDisplayState | None = None
    if config.display_backend == "web":
        web_server, web_state = start_web_display(
            config.web_host,
            config.web_port,
            config.input_interval_ms,
        )
        web_state.update_frame(latest_interactive_display)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, latest_interactive_display)

    logger.info(
        "Controls: w/s left paddle, i/k right paddle, r reset, q quit. "
        "Polling input every %.1f ms; no input is noop. Backend: %s.",
        config.input_interval_ms,
        config.display_backend,
    )

    tick_seconds = max(config.input_interval_ms, 0.0) / 1000.0
    next_tick_time = time.perf_counter()
    tick_idx = 0
    last_joint_action = 0
    last_mapped_action = -1
    try:
        with RawKeyboard() as keyboard:
            while True:
                tick_idx += 1
                tick_start_time = time.perf_counter()
                if config.display_backend == "web":
                    if web_state is None:
                        raise RuntimeError("Web display state was not initialized")
                    (
                        joint_action,
                        reset,
                        quit_requested,
                        input_post_count,
                        frame_publish_count,
                        frame_get_count,
                        stream_client_count,
                        stream_frame_count,
                    ) = web_state.consume_controls()
                    chars = keyboard.read_available()
                    terminal_action, terminal_reset, terminal_quit_requested = (
                        read_joint_action(chars)
                    )
                    reset = reset or terminal_reset
                    quit_requested = quit_requested or terminal_quit_requested
                    if terminal_action != 0:
                        joint_action = terminal_action
                else:
                    chars = keyboard.read_available()
                    cv_key = cv2.waitKey(1) & 0xFF
                    if cv_key != 255:
                        chars += chr(cv_key)
                    joint_action, reset, quit_requested = read_joint_action(chars)
                    input_post_count = 0
                    frame_publish_count = 0
                    frame_get_count = 0
                    stream_client_count = 0
                    stream_frame_count = 0

                if quit_requested:
                    logger.info("Quit requested at tick %s", tick_idx)
                    break
                if reset:
                    current_seed_window_index = sample_next_seed_window_index(
                        rng=seed_rng,
                        total_windows=seed_window_count,
                        current_index=current_seed_window_index,
                    )
                    rollout_state = initialize_interactive_rollout_state(
                        config=config,
                        action_model=action_model,
                        image_size=image_size,
                        interactive_device=interactive_device,
                        seed_window_index=current_seed_window_index,
                    )
                    generated_video = rollout_state.generated_video
                    source_action_history = rollout_state.source_action_history
                    latent_action_history = rollout_state.latent_action_history
                    latest_interactive_display = (
                        rollout_state.latest_interactive_display
                    )
                    joint_action = 0
                    last_joint_action = 0
                    last_mapped_action = -1
                    logger.info(
                        "Reset rolling window at tick %s with seed_window_index=%s",
                        tick_idx,
                        current_seed_window_index,
                    )
                    if config.display_backend == "web":
                        if web_state is None:
                            raise RuntimeError("Web display state was not initialized")
                        web_state.update_frame(latest_interactive_display)
                    else:
                        cv2.imshow(window_name, latest_interactive_display)

                    next_tick_time = max(next_tick_time + tick_seconds, tick_start_time)
                    sleep_seconds = next_tick_time - time.perf_counter()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    else:
                        next_tick_time = time.perf_counter()
                    continue

                if joint_action != last_joint_action:
                    logger.info(
                        "Tick %s input action changed: %s -> %s",
                        tick_idx,
                        last_joint_action,
                        joint_action,
                    )
                    last_joint_action = joint_action

                # A tick with no keypress intentionally appends action 0
                # (left noop, right noop) and still advances the model.
                source_action_history.append(joint_action)
                generation_start_time = time.perf_counter()
                (
                    next_frame,
                    mapped_action,
                    mapped_quality,
                    candidate_actions,
                ) = predict_next_interactive_frame(
                    action_mapping_model=action_mapping_model,
                    dynamics_model=dynamics_model,
                    rollout_tokenizer=tokenizer,
                    generated_video=generated_video,
                    source_action_history=source_action_history,
                    latent_action_history=latent_action_history,
                    mapping_sequence_length=mapping_sequence_length,
                    denoising_steps=config.denoising_steps,
                    use_amp=interactive_use_amp,
                )
                generation_ms = (time.perf_counter() - generation_start_time) * 1000.0
                if mapped_action != last_mapped_action:
                    logger.info(
                        "Tick %s mapped action changed: %s -> %s "
                        "(quality=%.2f candidates=%s)",
                        tick_idx,
                        last_mapped_action,
                        mapped_action,
                        mapped_quality,
                        candidate_actions,
                    )
                    last_mapped_action = mapped_action
                latent_action_history.append(mapped_action)
                generated_video = torch.cat(
                    [generated_video, next_frame.unsqueeze(1)],
                    dim=1,
                )
                (
                    generated_video,
                    source_action_history,
                    latent_action_history,
                ) = trim_rolling_state(
                    generated_video=generated_video,
                    source_action_history=source_action_history,
                    latent_action_history=latent_action_history,
                    rolling_window_length=rolling_window_length,
                )

                display_start_time = time.perf_counter()
                latest_interactive_display = frame_to_display(
                    generated_video[0, -1],
                    config.display_scale,
                )
                left_action = joint_action // 3
                right_action = joint_action % 3
                cv2.putText(
                    latest_interactive_display,
                    "Mapped controls | "
                    f"tick:{tick_idx} input L:{left_action} R:{right_action} "
                    f"mapped:{mapped_action} q:{mapped_quality:.1f}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    latest_interactive_display,
                    f"gen:{generation_ms:.1f}ms",
                    (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if config.display_backend == "web":
                    if web_state is None:
                        raise RuntimeError("Web display state was not initialized")
                    web_state.update_frame(latest_interactive_display)
                else:
                    cv2.imshow(window_name, latest_interactive_display)
                display_ms = (time.perf_counter() - display_start_time) * 1000.0

                tick_ms = (time.perf_counter() - tick_start_time) * 1000.0
                if (
                    config.timing_log_interval > 0
                    and tick_idx % config.timing_log_interval == 0
                ):
                    logger.info(
                        "Tick %s: input=%s mapped=%s quality=%.2f "
                        "gen=%.1fms display=%.1fms total=%.1fms "
                        "web_posts=%s web_published=%s web_gets=%s stream_clients=%s "
                        "stream_frames=%s candidates=%s window=%s",
                        tick_idx,
                        joint_action,
                        mapped_action,
                        mapped_quality,
                        generation_ms,
                        display_ms,
                        tick_ms,
                        input_post_count,
                        frame_publish_count,
                        frame_get_count,
                        stream_client_count,
                        stream_frame_count,
                        candidate_actions,
                        tuple(generated_video.shape),
                    )

                next_tick_time = max(next_tick_time + tick_seconds, tick_start_time)
                sleep_seconds = next_tick_time - time.perf_counter()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                else:
                    next_tick_time = time.perf_counter()
    finally:
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        if config.display_backend == "opencv":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(InteractiveInferenceConfig))
