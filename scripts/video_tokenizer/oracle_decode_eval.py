import logging
import math
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import tyro

from data.data_loaders.factory import build_datasets
from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.cache import Cache
from monitoring.codebook_usage import compute_codebook_usage
from monitoring.residual_coverage import compute_residual_coverage
from monitoring.videos import convert_video_to_images, save_comparison_images
from video_tokenization.checkpoints import load_model_from_checkpoint

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
    eval_samples: int = 500
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "tokenizer_oracle_decode_eval"
    max_comparison_batches: int = 2
    max_comparison_samples: int = 5


def _safe_psnr_from_mse(mse: float) -> float:
    if mse <= 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _accumulate_mse(total_sse: float, total_count: int, pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, int]:
    diff = (pred.float().clamp(0, 1) - gt.float().clamp(0, 1)) ** 2
    return total_sse + diff.sum().item(), total_count + diff.numel()


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
        getattr(tokenizer_config, "image_size", None),
        getattr(tokenizer_config, "patch_size", None),
        getattr(tokenizer_config, "num_images_in_video", None),
    )

    if config.local_cache_dir is None:
        raise ValueError("local_cache_dir is required")
    local_cache = Cache(
        max_size=config.max_cache_size,
        cache_dir=config.local_cache_dir,
    )
    train_dataset, test_dataset = build_datasets(
        config,
        local_cache,
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
    )

    logger.info(
        "Running oracle tokenizer decode on %s split: %d samples, batch_size=%d",
        config.eval_split,
        len(dataset),
        config.batch_size,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    direct_sse = 0.0
    codes_sse = 0.0
    direct_count = 0
    codes_count = 0
    decode_disagreement_sse = 0.0
    decode_disagreement_count = 0
    roundtrip_matches = 0
    roundtrip_total = 0
    code_tokens: list[torch.Tensor] = []
    residual_metrics: list[dict[str, float]] = []

    for batch_idx, video_batch in enumerate(dataloader):
        video_batch = video_batch.to(device, non_blocking=True)
        quantized = tokenizer.encode(video_batch)
        codes = tokenizer.quantized_value_to_codes(quantized)
        decoded_direct = tokenizer.decode(quantized)
        decoded_from_codes = tokenizer.decode_from_codes(codes)
        recoded = tokenizer.quantized_value_to_codes(tokenizer.fsq.indexes_to_codes(codes))

        direct_sse, direct_count = _accumulate_mse(
            direct_sse,
            direct_count,
            decoded_direct,
            video_batch,
        )
        codes_sse, codes_count = _accumulate_mse(
            codes_sse,
            codes_count,
            decoded_from_codes,
            video_batch,
        )
        decode_disagreement_sse, decode_disagreement_count = _accumulate_mse(
            decode_disagreement_sse,
            decode_disagreement_count,
            decoded_from_codes,
            decoded_direct,
        )

        roundtrip_matches += (recoded == codes).sum().item()
        roundtrip_total += codes.numel()
        code_tokens.append(codes.detach().cpu())

        if video_batch.shape[1] >= 2:
            residual_metrics.append(
                compute_residual_coverage(
                    gt_frame=video_batch[:, -1],
                    pred_frame=decoded_from_codes[:, -1],
                    prev_frame=video_batch[:, -2],
                )
            )

        if batch_idx < config.max_comparison_batches:
            n = min(config.max_comparison_samples, video_batch.shape[0])
            expected_videos = convert_video_to_images(video_batch[:n])
            predicted_videos = convert_video_to_images(decoded_from_codes[:n])
            image_path = os.path.join(
                config.output_dir,
                f"oracle_decode_batch_{batch_idx}.png",
            )
            save_comparison_images(predicted_videos, expected_videos, image_path)
            logger.info("Saved oracle decode comparison image to %s", image_path)

    direct_mse = direct_sse / max(direct_count, 1)
    codes_mse = codes_sse / max(codes_count, 1)
    decode_disagreement_mse = decode_disagreement_sse / max(decode_disagreement_count, 1)
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
