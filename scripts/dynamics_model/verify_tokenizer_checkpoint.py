import logging
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tyro

from data.data_loaders.video_window_loader import VideoWindowLoader
from data.datasets.atari_pong.atari_pong_dataset import AtariPongDataset
from data.datasets.atari_pong.atari_pong_dataset_creator import AtariPongDatasetCreator
from dynamics_model.training_args import DynamicsModelTrainingConfig
from video_tokenization.checkpoints import load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@dataclass
class TokenizerCheckpointVerificationConfig(DynamicsModelTrainingConfig):
    verification_batches: int = 1
    verification_num_workers: int = 2
    max_direct_vs_codes_mse: float = 1e-10


def _safe_psnr_from_mse(mse: float) -> float:
    if mse <= 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(
        pred.float().clamp(0, 1),
        target.float().clamp(0, 1),
        reduction="mean",
    ).item()


def _build_atari_test_dataset(
    config: TokenizerCheckpointVerificationConfig,
    required_eval_samples: int,
) -> AtariPongDataset:
    if config.dataset_type != "atari_pong":
        raise ValueError(
            "This verifier currently supports dataset_type='atari_pong'. "
            "The dynamics tokenizer load path is still shared with training."
        )
    if config.atari_pong_data_dir is None:
        raise ValueError("atari_pong_data_dir is required")

    creator = AtariPongDatasetCreator(
        data_dir=config.atari_pong_data_dir,
        num_frames_in_video=config.num_images_in_video,
        limit=config.dataset_limit,
        image_size=config.image_size,
        frame_spacing=config.frame_spacing,
        require_full_gameplay=config.atari_pong_require_full_gameplay,
    )
    _, raw_test = creator.setup()
    return AtariPongDataset(
        dataset=raw_test,
        image_size=config.image_size,
        num_images_in_video=config.num_images_in_video,
        crop_scoreboard=config.atari_pong_crop_scoreboard,
        limit=required_eval_samples,
    )


@torch.no_grad()
def verify_tokenizer_checkpoint(config: TokenizerCheckpointVerificationConfig) -> None:
    if not config.tokenizer_checkpoint_path:
        raise ValueError("tokenizer_checkpoint_path is required")

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

    first_parameter = next(tokenizer.parameters())
    logger.info(
        "Loaded tokenizer: vocab=%d, checkpoint_image_size=%d, checkpoint_patch_size=%d, "
        "checkpoint_T=%d, parameter_dtype=%s",
        tokenizer.get_vocab_size(),
        tokenizer_config.image_size,
        tokenizer_config.patch_size,
        tokenizer_config.num_images_in_video,
        first_parameter.dtype,
    )

    required_eval_samples = config.batch_size * config.verification_batches
    test_dataset = _build_atari_test_dataset(
        config,
        required_eval_samples,
    )
    dataloader = VideoWindowLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
        num_workers=config.verification_num_workers,
        seed=config.seed,
    )

    direct_mses: list[float] = []
    code_mses: list[float] = []
    direct_vs_codes_mses: list[float] = []
    roundtrip_matches = 0
    roundtrip_total = 0
    use_amp = config.use_bf16 and device.type == "cuda"

    for batch_idx, video_batch in enumerate(dataloader):
        if batch_idx >= config.verification_batches:
            break

        video_batch = video_batch.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            quantized = tokenizer.encode(video_batch)
            codes = tokenizer.quantized_value_to_codes(quantized)
            decoded_direct = tokenizer.decode(quantized)
            decoded_from_codes = tokenizer.decode_from_codes(codes)
            recoded = tokenizer.quantized_value_to_codes(
                tokenizer.fsq.indexes_to_codes(codes)
            )

        direct_mses.append(_mse(decoded_direct, video_batch))
        code_mses.append(_mse(decoded_from_codes, video_batch))
        direct_vs_codes_mses.append(_mse(decoded_from_codes, decoded_direct))
        roundtrip_matches += (recoded == codes).sum().item()
        roundtrip_total += codes.numel()

    if not direct_mses:
        raise RuntimeError("No verification batches were produced")

    direct_mse = sum(direct_mses) / len(direct_mses)
    code_mse = sum(code_mses) / len(code_mses)
    direct_vs_codes_mse = sum(direct_vs_codes_mses) / len(direct_vs_codes_mses)
    roundtrip_fraction = roundtrip_matches / max(roundtrip_total, 1)

    logger.info(
        "Tokenizer verification complete: direct_mse=%.10f (psnr=%.4f), "
        "codes_mse=%.10f (psnr=%.4f), direct_vs_codes_mse=%.12f, "
        "roundtrip=%d/%d (%.6f%%), autocast=%s",
        direct_mse,
        _safe_psnr_from_mse(direct_mse),
        code_mse,
        _safe_psnr_from_mse(code_mse),
        direct_vs_codes_mse,
        roundtrip_matches,
        roundtrip_total,
        100.0 * roundtrip_fraction,
        use_amp,
    )

    if direct_vs_codes_mse > config.max_direct_vs_codes_mse:
        raise RuntimeError(
            "decode_from_codes does not match direct decode: "
            f"mse={direct_vs_codes_mse:.12f}"
        )
    if roundtrip_matches != roundtrip_total:
        raise RuntimeError(
            "Code roundtrip mismatch: "
            f"{roundtrip_matches}/{roundtrip_total} tokens matched"
        )


def main() -> None:
    config = tyro.cli(TokenizerCheckpointVerificationConfig)
    verify_tokenizer_checkpoint(config)


if __name__ == "__main__":
    main()
