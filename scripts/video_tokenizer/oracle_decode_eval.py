import logging
import math
import os
from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.residual_coverage import compute_residual_coverage
from monitoring.videos import convert_video_to_images
from video_tokenization.checkpoints import load_model_from_checkpoint
from video_tokenization.model import VideoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@dataclass
class OracleDecodeEvalConfig:
    tokenizer_checkpoint_path: str
    dataset_type: Literal["pokemon", "atari_pong"] = "atari_pong"
    atari_pong_data_dir: Optional[str] = "data/atari_pong"
    atari_pong_crop_scoreboard: bool = False
    atari_pong_require_full_gameplay: bool = True
    frames_dir: str = "pokemon_frames"
    dataset_train_key: Optional[str] = None
    sync_from_s3: bool = False
    use_s3: bool = False
    local_cache_dir: Optional[str] = os.environ.get("BT_RW_CACHE_DIR", "cache")
    max_cache_size: int = 100000
    image_size: int = 84
    patch_size: int = 4
    num_images_in_video: int = 5
    frame_spacing: int = 1
    num_unique_frames: Optional[int] = None
    dataset_limit: int = 1_000_000
    eval_split: Literal["train", "test"] = "test"
    eval_samples: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "tokenizer_oracle_decode_eval"
    max_comparison_batches: int = 1
    max_comparison_samples: int = 1
    reconstruction_error_scale: float = 5.0
    autoregressive_rollout_frames: int = 20


def _safe_psnr_from_mse(mse: float) -> float:
    if mse <= 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _accumulate_mse(
    total_sse: float,
    total_count: int,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[float, int]:
    diff = (pred.float().clamp(0, 1) - gt.float().clamp(0, 1)) ** 2
    return total_sse + diff.sum().item(), total_count + diff.numel()


def _make_frame_triptych(
    expected_image: Image.Image,
    predicted_image: Image.Image,
    error_image: Image.Image,
) -> Image.Image:
    images = [
        expected_image.convert("RGB"),
        predicted_image.convert("RGB"),
        error_image.convert("RGB"),
    ]
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    triptych = Image.new("RGB", (width, height), color=(0, 0, 0))

    x_offset = 0
    for image in images:
        triptych.paste(image, (x_offset, 0))
        x_offset += image.width

    return triptych


def _save_oracle_decode_rollout_grid(
    predicted_videos: list[list[Image.Image]],
    expected_videos: list[list[Image.Image]],
    error_videos: list[list[Image.Image]],
    output_path: str,
) -> str:
    if not predicted_videos:
        raise ValueError("Cannot save rollout grid for an empty batch")

    rows = 4
    cols = 5
    num_grid_frames = rows * cols
    sequence_idx = 0
    num_frames = min(len(predicted_videos[sequence_idx]), num_grid_frames)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.0))
    flat_axes = np.asarray(axs).reshape(-1)
    for frame_idx, ax in enumerate(flat_axes):
        ax.axis("off")
        if frame_idx >= num_frames:
            continue

        triptych = _make_frame_triptych(
            expected_videos[sequence_idx][frame_idx],
            predicted_videos[sequence_idx][frame_idx],
            error_videos[sequence_idx][frame_idx],
        )
        ax.imshow(triptych, interpolation="nearest")
        ax.set_title(f"Frame {frame_idx}")

    fig.suptitle("Each cell: Orig | Pred | Abs Err", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _encode_decode_window(
    tokenizer: VideoTokenizer,
    video_window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized = tokenizer.encode(video_window)
    codes = tokenizer.quantized_value_to_codes(quantized)
    decoded_direct = tokenizer.decode(quantized)
    decoded_from_codes = tokenizer.decode_from_codes(codes)
    return decoded_direct, decoded_from_codes, codes


def _build_rolling_tokenizer_window(
    video_batch: torch.Tensor,
    context_source: torch.Tensor,
    frame_idx: int,
    context_frames: int,
) -> torch.Tensor:
    previous_context_frames = context_frames - 1
    start_idx = max(0, frame_idx - previous_context_frames)
    history = context_source[:, start_idx:frame_idx]
    pad_frames = previous_context_frames - history.shape[1]

    if pad_frames > 0:
        seed_padding = video_batch[:, :1].expand(-1, pad_frames, -1, -1, -1)
        history = torch.cat([seed_padding, history], dim=1)

    target_frame = video_batch[:, frame_idx : frame_idx + 1]
    return torch.cat([history, target_frame], dim=1)


def _autoregressive_reconstruct(
    tokenizer: VideoTokenizer,
    video_batch: torch.Tensor,
    context_frames: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    seed_frames = 1
    if video_batch.shape[1] <= seed_frames:
        raise ValueError(
            f"Need more than {seed_frames} frame for autoregressive rollout, "
            f"got {video_batch.shape[1]}"
        )

    reconstructed = torch.empty_like(video_batch)
    code_tokens: list[torch.Tensor] = []
    reconstructed[:, :seed_frames] = video_batch[:, :seed_frames]

    for frame_idx in range(seed_frames, video_batch.shape[1]):
        context_window = _build_rolling_tokenizer_window(
            video_batch,
            reconstructed,
            frame_idx,
            context_frames,
        )
        _, decoded_from_codes, codes = _encode_decode_window(tokenizer, context_window)
        reconstructed[:, frame_idx] = decoded_from_codes[:, -1]
        code_tokens.append(codes.detach().cpu())

    return reconstructed, code_tokens


def _non_autoregressive_reconstruct(
    tokenizer: VideoTokenizer,
    video_batch: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    seed_frames = 1
    if video_batch.shape[1] <= seed_frames:
        raise ValueError(
            f"Need more than {seed_frames} frame for rollout, "
            f"got {video_batch.shape[1]}"
        )

    reconstructed = torch.empty_like(video_batch)
    reconstructed[:, :seed_frames] = video_batch[:, :seed_frames]

    for frame_idx in range(seed_frames, video_batch.shape[1]):
        context_window = _build_rolling_tokenizer_window(
            video_batch,
            video_batch,
            frame_idx,
            context_frames,
        )
        _, decoded_from_codes, _ = _encode_decode_window(tokenizer, context_window)
        reconstructed[:, frame_idx] = decoded_from_codes[:, -1]

    return reconstructed


def _accumulate_per_frame_sse(
    total_sse: torch.Tensor,
    total_count: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (pred.float().clamp(0, 1) - gt.float().clamp(0, 1)) ** 2
    frame_sse = diff.sum(dim=(0, 2, 3, 4)).detach().cpu()
    frame_count = torch.full(
        (diff.shape[1],),
        diff.shape[0] * diff.shape[2] * diff.shape[3] * diff.shape[4],
        dtype=torch.float64,
    )
    return total_sse + frame_sse.double(), total_count + frame_count


def _save_autoregressive_metrics_plot(
    autoregressive_mse_by_frame: list[float],
    non_autoregressive_mse_by_frame: list[float],
    output_path: str,
    first_frame_idx: int,
) -> str:
    frames = list(
        range(first_frame_idx, first_frame_idx + len(autoregressive_mse_by_frame))
    )
    min_positive_mse = 1e-12
    autoregressive_mse_for_plot = [
        max(mse, min_positive_mse) for mse in autoregressive_mse_by_frame
    ]
    non_autoregressive_mse_for_plot = [
        max(mse, min_positive_mse) for mse in non_autoregressive_mse_by_frame
    ]
    autoregressive_psnr = [
        _safe_psnr_from_mse(mse) for mse in autoregressive_mse_by_frame
    ]
    non_autoregressive_psnr = [
        _safe_psnr_from_mse(mse) for mse in non_autoregressive_mse_by_frame
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(frames, non_autoregressive_mse_for_plot, label="Non-autoregressive")
    axs[0].plot(frames, autoregressive_mse_for_plot, label="Autoregressive")
    axs[0].set_title("Per-frame MSE")
    axs[0].set_xlabel("Frame")
    axs[0].set_ylabel("MSE")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    axs[0].grid(True, alpha=0.3)
    axs[0].grid(True, which="minor", alpha=0.15)
    axs[0].legend()

    axs[1].plot(frames, non_autoregressive_psnr, label="Non-autoregressive")
    axs[1].plot(frames, autoregressive_psnr, label="Autoregressive")
    axs[1].set_title("Per-frame PSNR")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("PSNR")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


@torch.no_grad()
def evaluate_oracle_decode(config: OracleDecodeEvalConfig) -> None:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    logger.info("Loading tokenizer from %s", config.tokenizer_checkpoint_path)
    tokenizer, tokenizer_config = load_model_from_checkpoint(
        config.tokenizer_checkpoint_path,
        device,
    )
    tokenizer.eval()
    logger.info(
        "Loaded tokenizer: vocab=%d, checkpoint_image_size=%s, checkpoint_patch_size=%s, checkpoint_T=%s",
        tokenizer.get_vocab_size(),
        tokenizer_config.image_size,
        tokenizer_config.patch_size,
        tokenizer_config.num_images_in_video,
    )

    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")
    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )
    tokenizer_context_frames = tokenizer_config.num_images_in_video
    eval_sequence_frames = max(
        config.autoregressive_rollout_frames,
        tokenizer_context_frames,
    )
    train_dataset, test_dataset = build_datasets(
        config,
        local_cache,
        num_frames_in_video=eval_sequence_frames,
        test_limit=config.eval_samples,
    )
    dataset = train_dataset if config.eval_split == "train" else test_dataset
    dataloader = VideoWindowLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        drop_last=False,
    )

    logger.info(
        "Running pseudo-autoregressive tokenizer decode on %s split: "
        "%d samples, batch_size=%d, context_frames=%d, rollout_frames=%d, "
        "scored_frame_start=%d",
        config.eval_split,
        len(dataset),
        config.batch_size,
        tokenizer_context_frames,
        eval_sequence_frames,
        1,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    scored_frame_start = 1
    scored_frame_count = eval_sequence_frames - scored_frame_start
    direct_sse = 0.0
    codes_sse = 0.0
    direct_count = 0
    codes_count = 0
    decode_disagreement_sse = 0.0
    decode_disagreement_count = 0
    autoregressive_sse = 0.0
    autoregressive_count = 0
    non_autoregressive_sse = 0.0
    non_autoregressive_count = 0
    autoregressive_frame_sse = torch.zeros(scored_frame_count, dtype=torch.float64)
    autoregressive_frame_count = torch.zeros(scored_frame_count, dtype=torch.float64)
    non_autoregressive_frame_sse = torch.zeros(
        scored_frame_count,
        dtype=torch.float64,
    )
    non_autoregressive_frame_count = torch.zeros(
        scored_frame_count,
        dtype=torch.float64,
    )
    roundtrip_matches = 0
    roundtrip_total = 0
    code_tokens: list[torch.Tensor] = []
    residual_metrics: list[dict[str, float]] = []

    for batch_idx, video_batch in enumerate(dataloader):
        video_batch = video_batch.to(device, non_blocking=True)
        oracle_window = video_batch[:, :tokenizer_context_frames]
        decoded_direct, decoded_from_codes, codes = _encode_decode_window(
            tokenizer,
            oracle_window,
        )
        autoregressive_decoded, autoregressive_code_tokens = (
            _autoregressive_reconstruct(
                tokenizer,
                video_batch,
                tokenizer_context_frames,
            )
        )
        non_autoregressive_decoded = _non_autoregressive_reconstruct(
            tokenizer,
            video_batch,
            tokenizer_context_frames,
        )
        recoded = tokenizer.quantized_value_to_codes(
            tokenizer.fsq.indexes_to_codes(codes)
        )

        direct_sse, direct_count = _accumulate_mse(
            direct_sse,
            direct_count,
            decoded_direct,
            oracle_window,
        )
        codes_sse, codes_count = _accumulate_mse(
            codes_sse,
            codes_count,
            decoded_from_codes,
            oracle_window,
        )
        decode_disagreement_sse, decode_disagreement_count = _accumulate_mse(
            decode_disagreement_sse,
            decode_disagreement_count,
            decoded_from_codes,
            decoded_direct,
        )
        autoregressive_sse, autoregressive_count = _accumulate_mse(
            autoregressive_sse,
            autoregressive_count,
            autoregressive_decoded[:, scored_frame_start:],
            video_batch[:, scored_frame_start:],
        )
        non_autoregressive_sse, non_autoregressive_count = _accumulate_mse(
            non_autoregressive_sse,
            non_autoregressive_count,
            non_autoregressive_decoded[:, scored_frame_start:],
            video_batch[:, scored_frame_start:],
        )
        autoregressive_frame_sse, autoregressive_frame_count = (
            _accumulate_per_frame_sse(
                autoregressive_frame_sse,
                autoregressive_frame_count,
                autoregressive_decoded[:, scored_frame_start:],
                video_batch[:, scored_frame_start:],
            )
        )
        non_autoregressive_frame_sse, non_autoregressive_frame_count = (
            _accumulate_per_frame_sse(
                non_autoregressive_frame_sse,
                non_autoregressive_frame_count,
                non_autoregressive_decoded[:, scored_frame_start:],
                video_batch[:, scored_frame_start:],
            )
        )

        roundtrip_matches += (recoded == codes).sum().item()
        roundtrip_total += codes.numel()
        code_tokens.extend(autoregressive_code_tokens)

        if video_batch.shape[1] >= 2:
            residual_metrics.append(
                compute_residual_coverage(
                    gt_frame=video_batch[:, -1],
                    pred_frame=autoregressive_decoded[:, -1],
                    prev_frame=video_batch[:, -2],
                )
            )

        if batch_idx < config.max_comparison_batches:
            n = min(config.max_comparison_samples, video_batch.shape[0])
            expected_videos = convert_video_to_images(video_batch[:n])
            predicted_videos = convert_video_to_images(autoregressive_decoded[:n])
            error_videos = convert_video_to_images(
                autoregressive_decoded[:n] - video_batch[:n],
                value_mode="magnitude",
                residual_scale=config.reconstruction_error_scale,
            )
            image_path = os.path.join(
                config.output_dir,
                f"oracle_decode_batch_{batch_idx}.png",
            )
            _save_oracle_decode_rollout_grid(
                predicted_videos,
                expected_videos,
                error_videos,
                image_path,
            )
            logger.info("Saved oracle decode comparison image to %s", image_path)

    direct_mse = direct_sse / max(direct_count, 1)
    codes_mse = codes_sse / max(codes_count, 1)
    decode_disagreement_mse = decode_disagreement_sse / max(decode_disagreement_count, 1)
    autoregressive_mse = autoregressive_sse / max(autoregressive_count, 1)
    non_autoregressive_mse = non_autoregressive_sse / max(
        non_autoregressive_count,
        1,
    )
    autoregressive_mse_by_frame = (
        autoregressive_frame_sse / torch.clamp(autoregressive_frame_count, min=1)
    ).tolist()
    non_autoregressive_mse_by_frame = (
        non_autoregressive_frame_sse
        / torch.clamp(non_autoregressive_frame_count, min=1)
    ).tolist()
    metrics_plot_path = os.path.join(
        config.output_dir,
        "autoregressive_vs_non_autoregressive_metrics.png",
    )
    _save_autoregressive_metrics_plot(
        autoregressive_mse_by_frame,
        non_autoregressive_mse_by_frame,
        metrics_plot_path,
        scored_frame_start,
    )
    logger.info("Saved autoregressive metrics plot to %s", metrics_plot_path)
    usage_tokens = (
        torch.cat([tokens.reshape(-1) for tokens in code_tokens], dim=0)
        if code_tokens
        else torch.empty(0, dtype=torch.long)
    )
    codebook_usage = compute_codebook_usage(usage_tokens, tokenizer.get_vocab_size())

    logger.info("Oracle tokenizer decode results")
    logger.info(
        "direct decode: mse=%.8f, psnr=%.4f",
        direct_mse,
        _safe_psnr_from_mse(direct_mse),
    )
    logger.info(
        "decode_from_codes: mse=%.8f, psnr=%.4f",
        codes_mse,
        _safe_psnr_from_mse(codes_mse),
    )
    logger.info(
        "pseudo-autoregressive decode: mse=%.8f, psnr=%.4f",
        autoregressive_mse,
        _safe_psnr_from_mse(autoregressive_mse),
    )
    logger.info(
        "non-autoregressive rolling decode: mse=%.8f, psnr=%.4f",
        non_autoregressive_mse,
        _safe_psnr_from_mse(non_autoregressive_mse),
    )
    logger.info(
        "direct_vs_codes: mse=%.10f, psnr=%.4f",
        decode_disagreement_mse,
        _safe_psnr_from_mse(decode_disagreement_mse),
    )
    logger.info(
        "code roundtrip: %d/%d tokens match (%.6f%%)",
        roundtrip_matches,
        roundtrip_total,
        100.0 * roundtrip_matches / max(roundtrip_total, 1),
    )
    logger.info(
        "codebook usage: unique=%d/%d (%.2f%%), perplexity=%.2f, norm_entropy=%.4f",
        int(codebook_usage["num_unique"]),
        tokenizer.get_vocab_size(),
        100.0 * codebook_usage["usage_fraction"],
        codebook_usage["perplexity"],
        codebook_usage["normalized_entropy"],
    )

    if residual_metrics:
        keys = residual_metrics[0].keys()
        averaged = {
            key: sum(metrics[key] for metrics in residual_metrics) / len(residual_metrics)
            for key in keys
        }
        copy_prev_psnr = _safe_psnr_from_mse(averaged["copy_prev_mse"])
        oracle_psnr = _safe_psnr_from_mse(averaged["pred_mse"])
        logger.info(
            "final-frame residual coverage: R2=%.4f, cosine=%.4f, "
            "pred_mse=%.8f, pred_psnr=%.4f, copy_prev_mse=%.8f, copy_prev_psnr=%.4f, "
            "changed_px_mse=%.8f, changed_frac=%.4f",
            averaged["residual_r2"],
            averaged["residual_cosine"],
            averaged["pred_mse"],
            oracle_psnr,
            averaged["copy_prev_mse"],
            copy_prev_psnr,
            averaged["changed_pixel_mse"],
            averaged["changed_pixel_fraction"],
        )


def main() -> None:
    config = tyro.cli(OracleDecodeEvalConfig)
    evaluate_oracle_decode(config)


if __name__ == "__main__":
    main()
