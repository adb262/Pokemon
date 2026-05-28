from typing import Literal

from einops.einops import rearrange
import torch
import torch.nn as nn
from torch.distributions import uniform

from latent_action_model.model import LatentActionVQVAE
from loss.loss_fns import (
    clipped_cross_entropy_loss,
    clipped_next_frame_reconstruction_loss,
    clipped_next_frame_reconstruction_residual_loss,
    next_frame_reconstruction_loss,
    next_frame_reconstruction_residual_loss,
)
from torch_utilities.initialize import init_weights
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer
from video_tokenization.model import VideoTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        mask_ratio_lower_bound: float,
        mask_ratio_upper_bound: float,
        num_images_in_video: int,
        num_heads: int,
        num_layers: int,
        d_model: int,
        tokenizer: VideoTokenizer,
        action_model: LatentActionVQVAE,
        predict_action_residuals: bool,
        action_decoder_loss: Literal["l2", "clipped_l2"],
        action_l2_clip_c: float,
        dynamics_token_loss: Literal["ce", "clipped_ce"],
        dynamics_ce_clip_c: float,
    ):
        super(DynamicsModel, self).__init__()
        self.mask_ratio_lower_bound = mask_ratio_lower_bound
        self.mask_ratio_upper_bound = mask_ratio_upper_bound
        self.num_images_in_video = num_images_in_video
        self.d_model = d_model
        self.predict_action_residuals = predict_action_residuals
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_ratio_distribution = uniform.Uniform(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )

        self.tokenizer_vocab_size = tokenizer.get_vocab_size()
        self.action_vocab_size = action_model.action_vocab_size

        self.tokenizer = tokenizer
        self.action_model = action_model
        self._ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.dynamics_token_loss = dynamics_token_loss
        self.dynamics_ce_clip_c = dynamics_ce_clip_c

        # python -m scripts.video_tokenizer.train --frames_dir pokemon_frames/pokemon_emerald --num_images_in_video 5 --batch_size 16 --save_dir dynamics_model_base --bins 8 8 6 5 --use_s3 --dataset_train_key pokemon_emerald_train_0_9_5_frames.json --checkpoint_dir dynamics_model_base --patch_size 4 --image_size 128 --num_epochs 20 --gradient_clipping 1.0 --tokenizer_checkpoint_path fsq_tokenizer_2k_128_4_512_8_heads_4_layers

        # +1 for the mask token
        self.tokenizer_embedding = nn.Embedding(self.tokenizer_vocab_size + 1, d_model)
        self.action_embedding = nn.Embedding(self.action_vocab_size + 1, d_model)
        self.decoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.vocab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, self.tokenizer.get_vocab_size()),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.changed_patch_loss_weight = 30.0
        self.action_loss_fn = self._build_action_loss_fn(
            predict_action_residuals=predict_action_residuals,
            action_decoder_loss=action_decoder_loss,
            action_l2_clip_c=action_l2_clip_c,
        )

        # Only initialize the newly-created submodules. ``self.apply`` would
        # recurse into ``self.tokenizer`` and ``self.action_model`` and wipe
        # their pretrained / co-trained weights.
        for module in (
            self.tokenizer_embedding,
            self.action_embedding,
            self.decoder,
            self.vocab_head,
        ):
            module.apply(init_weights)

    @staticmethod
    def _build_action_loss_fn(
        *,
        predict_action_residuals: bool,
        action_decoder_loss: Literal["l2", "clipped_l2"],
        action_l2_clip_c: float,
    ):
        if action_decoder_loss == "l2":
            return (
                next_frame_reconstruction_residual_loss
                if predict_action_residuals
                else next_frame_reconstruction_loss
            )
        if action_decoder_loss == "clipped_l2":
            clipped_loss_fn = (
                clipped_next_frame_reconstruction_residual_loss
                if predict_action_residuals
                else clipped_next_frame_reconstruction_loss
            )

            def action_loss(video: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
                return clipped_loss_fn(
                    video,
                    decoded,
                    l2_clip_c=action_l2_clip_c,
                )

            return action_loss
        raise ValueError(f"Unknown action_decoder_loss: {action_decoder_loss!r}")

    @staticmethod
    def cosine_scheduler(max_steps: int, current_step: int, device: torch.device):
        return 0.5 * (1 + torch.cos(torch.pi * torch.tensor(current_step) / torch.tensor(max_steps, device=device)))

    def forward(
        self, video: torch.Tensor, video_prediction_basis: torch.Tensor, action_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """video_prediction_basis is the same shape as video, but is the basis for the prediction. video is only used for the loss computation.
        
        This allows the caller to tell us to use previous predictions as basis."""
        # input is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # Return our predictions (batch_size, num_images_in_video, num_patches, vocab_size)
        # And the masked targets

        assert video.shape == video_prediction_basis.shape, "video and video_prediction_basis must have the same shape"

        logger.debug(f"video shape: {video.shape}")
        # x is of shape (batch_size, num_images_in_video, num_patches)
        with torch.no_grad():
            targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video_prediction_basis)
            ).long()
            original_targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video)
            ).long()

        batch_size, _, num_patches = targets.shape
        mask_ratio = torch.empty(batch_size, 1, 1, device=targets.device).uniform_(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )
        mask = (
            torch.rand(batch_size, self.num_images_in_video - 1, num_patches, device=targets.device) < mask_ratio
        )  # Boolean mask over only the target frame

        targets[:, 1:, :] = self.tokenizer.mask_codebook_tokens(
            targets[:, 1:, :].long(), mask
        )

        # targets is of shape (batch_size, num_images_in_video, num_patches)
        x = self.tokenizer_embedding(targets.long())
        action_embeddings = self.action_embedding(action_tokens.long())
        logger.debug(f"action_embeddings shape: {action_embeddings.shape}")
        logger.debug(f"x shape: {x.shape}")

        # Unsqueeze to add to each patch in the sequence
        x[:, :-1, :] += action_embeddings.unsqueeze(2)

        x = self.decoder(x)
        x = self.vocab_head(x)

        # Only compute token loss on masked positions of the final frame.
        target_frame = original_targets[:, 1:, :].masked_fill(~mask, -100)

        # View both as 3d tensors for the loss function
        logger.debug(f"x shape: {x.shape}, targets shape: {targets.shape}")
        # predicted_tokens = rearrange(x[:, -1:, :, :], "b t p d -> b (t p) d")
        # target_tokens = rearrange(target_frame, "b t p -> b (t p)")
        # logger.debug(f"predicted_tokens shape: {predicted_tokens.shape}, target_tokens shape: {target_tokens.shape}")
        predicted_tokens = rearrange(x[:, 1:, :, :], "b t p d -> b (t p) d")
        target_tokens = rearrange(target_frame, "b t p -> b (t p)")
        if self.dynamics_token_loss == "ce":
            token_loss = self._ce_loss_fn(
                predicted_tokens.transpose(1, 2), target_tokens
            )
        elif self.dynamics_token_loss == "clipped_ce":
            token_loss = clipped_cross_entropy_loss(
                predicted_tokens=predicted_tokens,
                target_tokens=target_tokens,
                ce_clip_c=self.dynamics_ce_clip_c,
            )
        else:
            raise ValueError(f"Unknown dynamics_token_loss: {self.dynamics_token_loss!r}")

        return x, token_loss, action_tokens
    
    @torch.no_grad()
    def predict_next_frame(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        max_steps: int = 10,
        decode_tokenizer: VideoTokenizer | None = None,
        context_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next frame after the observed context.

        ``context`` may contain one or more frames. Dynamics conditioning is
        trimmed to the most recent ``T - 1`` frames, while decoding uses the
        tokenizer checkpoint's own rolling frame window.
        """
        max_context = self.num_images_in_video - 1
        num_frames = context.shape[1]
        if num_frames < 1:
            raise ValueError(
                "predict_next_frame requires at least 1 context frame, "
                f"got {num_frames}"
            )

        context_window = context[:, -max_context:]
        next_codes = self.predict_next_frame_codes(
            context_window,
            action,
            max_steps=max_steps,
            context_actions=context_actions,
        )
        return self.decode_next_frame_with_tokenizer_window(
            context,
            next_codes,
            decode_tokenizer=decode_tokenizer,
        )

    @torch.no_grad()
    def predict_next_frame_codes(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        max_steps: int = 10,
        context_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next-frame tokenizer codes without decoding the dynamics window."""
        max_context = self.num_images_in_video - 1
        num_frames = context.shape[1]
        if num_frames < 1:
            raise ValueError(
                "predict_next_frame_codes requires at least 1 context frame, "
                f"got {num_frames}"
            )

        context_window = context[:, -max_context:]
        placeholder = context_window[:, -1:].clone()
        inference_window = torch.cat([context_window, placeholder], dim=1)

        targets = self.tokenizer.quantized_value_to_codes(
            self.tokenizer.encode(inference_window)
        ).long()
        batch_size, _, num_patches = targets.shape

        action_token = action.long().to(targets.device)
        if action_token.dim() == 1:
            action_token = action_token.unsqueeze(1)

        targets[:, -1, :] = self.tokenizer.get_mask_token_idx()

        expected_context = inference_window.shape[1] - 2
        if context_actions is not None:
            context_action_tokens = context_actions.long().to(targets.device)
            if context_action_tokens.dim() == 1:
                if batch_size != 1:
                    raise ValueError(
                        "context_actions with shape (K,) is only valid for "
                        f"batch size 1, got {batch_size}"
                    )
                context_action_tokens = context_action_tokens.unsqueeze(0)
            if context_action_tokens.dim() != 2:
                raise ValueError(
                    "context_actions must have shape (B, K); got tensor with "
                    f"shape {tuple(context_action_tokens.shape)}"
                )
            if context_action_tokens.shape[0] != batch_size:
                raise ValueError(
                    "context_actions batch dimension must match context batch "
                    f"size: {context_action_tokens.shape[0]} vs {batch_size}"
                )
            if context_action_tokens.shape[1] < expected_context:
                raise ValueError(
                    "context_actions must cover every transition inside the "
                    f"context window: got {context_action_tokens.shape[1]}, "
                    f"need {expected_context}"
                )
            if expected_context == 0:
                context_action_tokens = torch.empty(
                    batch_size, 0, dtype=torch.long, device=targets.device
                )
            else:
                context_action_tokens = context_action_tokens[:, -expected_context:]
        elif expected_context > 0:
            action_video_encoded = self.action_model.encode(inference_window[:, :-1])
            context_action_tokens = self.action_model.get_action_sequence(
                action_video_encoded
            ).long()
        else:
            context_action_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=targets.device
            )

        action_tokens = torch.cat([context_action_tokens, action_token], dim=1)
        action_embeddings = self.action_embedding(action_tokens.long())

        step = 0
        mask_locations = torch.full(
            (batch_size, num_patches), True, dtype=torch.bool, device=targets.device
        )
        while step < max_steps:
            x = self.tokenizer_embedding(targets.long())
            x[:, :-1, :] += action_embeddings.unsqueeze(2)
            x = self.decoder(x)
            logits = self.vocab_head(x)

            if not mask_locations.any():
                break

            ratio_remaining = self.cosine_scheduler(max_steps, step + 1, targets.device)
            target_still_masked = int(num_patches * ratio_remaining)
            current_masked = int(mask_locations.sum(dim=1).amax().item())
            tokens_to_update = max(0, current_masked - target_still_masked)

            probs = self.softmax(logits[:, -1])
            mask_flat = mask_locations.view(-1)
            probs_flat = probs.view(-1, probs.size(-1))
            samples_flat = torch.full(
                (batch_size * num_patches,),
                -100,
                device=targets.device,
                dtype=torch.long,
            )
            if mask_flat.any():
                probs_masked = probs_flat[mask_flat]
                probs_masked = probs_masked / (
                    probs_masked.sum(-1, keepdim=True) + 1e-9
                )
                sampled_masked = torch.multinomial(
                    probs_masked, 1, replacement=False
                ).squeeze(-1)
                samples_flat[mask_flat] = sampled_masked

            samples = samples_flat.view(batch_size, num_patches)
            sampled_scores = probs.gather(
                -1, samples.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            sampled_scores[samples < 0] = -1

            if tokens_to_update > 0:
                _, top_positions = torch.topk(
                    sampled_scores, tokens_to_update, dim=-1
                )
                top_tokens = samples.gather(1, top_positions)

                targets_last = targets[:, -1, :]
                targets_last = targets_last.scatter(1, top_positions, top_tokens)
                targets[:, -1, :] = targets_last

                mask_locations = mask_locations.scatter(
                    1, top_positions, torch.zeros_like(top_positions, dtype=torch.bool)
                )

            step += 1

        return targets[:, -1, :]

    def build_tokenizer_decode_history(
        self,
        generated_pixels: torch.Tensor,
        decode_tokenizer: VideoTokenizer | None = None,
    ) -> torch.Tensor:
        tokenizer = self.tokenizer if decode_tokenizer is None else decode_tokenizer
        tokenizer_window = tokenizer.decoder.num_images_in_video
        if tokenizer_window < 2:
            raise ValueError(
                "Tokenizer decoder window must be at least 2 frames for rollout, "
                f"got {tokenizer_window}"
            )

        previous_context_frames = tokenizer_window - 1
        history = generated_pixels[:, -previous_context_frames:]
        pad_frames = previous_context_frames - history.shape[1]
        if pad_frames > 0:
            seed_padding = generated_pixels[:, :1].expand(
                -1, pad_frames, -1, -1, -1
            )
            history = torch.cat([seed_padding, history], dim=1)
        return history

    @torch.no_grad()
    def decode_next_frame_with_tokenizer_window(
        self,
        generated_pixels: torch.Tensor,
        next_codes: torch.Tensor,
        decode_tokenizer: VideoTokenizer | None = None,
    ) -> torch.Tensor:
        tokenizer = self.tokenizer if decode_tokenizer is None else decode_tokenizer
        tokenizer_history = self.build_tokenizer_decode_history(
            generated_pixels,
            decode_tokenizer=tokenizer,
        )
        quantized_context = tokenizer.encode(tokenizer_history)
        context_codes = tokenizer.quantized_value_to_codes(quantized_context)
        all_codes = torch.cat([context_codes, next_codes.unsqueeze(1)], dim=1)
        decoded_window = tokenizer.decode_from_codes(all_codes)
        return decoded_window[:, -1]

    def normalize_rollout_actions(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        action_sequence = actions.long().to(video.device)
        if action_sequence.dim() == 1:
            if video.shape[0] != 1:
                raise ValueError(
                    "rollout expects actions with shape (B, K); got a 1D tensor "
                    f"for batch size {video.shape[0]}"
                )
            action_sequence = action_sequence.unsqueeze(0)
        if action_sequence.dim() != 2:
            raise ValueError(
                "rollout expects actions with shape (B, K); got tensor with "
                f"shape {tuple(action_sequence.shape)}"
            )
        if action_sequence.shape[0] != video.shape[0]:
            raise ValueError(
                "rollout actions batch dimension must match video batch size: "
                f"{action_sequence.shape[0]} vs {video.shape[0]}"
            )
        return action_sequence

    @torch.no_grad()
    def rollout(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        max_steps: int = 10,
        decode_tokenizer: VideoTokenizer | None = None,
        seed_context_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate future frames autoregressively from an initial context.

        Given a seed ``video`` of shape ``(B, S, C, H, W)`` and ``actions``
        of shape ``(B, K)``, produces ``(B, S + K, C, H, W)`` by generating
        one new frame per action. ``S`` may be as small as ``1``.

        At each step, the dynamics model sees at most ``num_images_in_video - 1``
        prior frames plus a masked next-frame placeholder, matching training's
        ``num_images_in_video`` window. The decode tokenizer sees its own
        rolling decoder window, which may be shorter than the dynamics window.
        Seed context action tokens are provided by the caller or inferred once
        from the seed video, then preserved alongside rollout actions.
        """
        original_num_frames = video.shape[1]
        if original_num_frames < 1:
            raise ValueError(
                f"rollout requires at least 1 seed frame, got {original_num_frames}"
            )

        action_sequence = self.normalize_rollout_actions(video, actions)
        num_rollout_steps = action_sequence.shape[1]
        max_context = self.num_images_in_video - 1
        seed_action_sequence: torch.Tensor | None = None
        if seed_context_actions is not None:
            seed_action_sequence = self.normalize_rollout_actions(
                video,
                seed_context_actions,
            )
            expected_seed_actions = original_num_frames - 1
            if seed_action_sequence.shape[1] < expected_seed_actions:
                raise ValueError(
                    "seed_context_actions must include every transition inside "
                    f"the seed video: got {seed_action_sequence.shape[1]}, "
                    f"need {expected_seed_actions}"
                )
            if expected_seed_actions == 0:
                seed_action_sequence = torch.empty(
                    video.shape[0], 0, dtype=torch.long, device=video.device
                )
            else:
                seed_action_sequence = seed_action_sequence[:, -expected_seed_actions:]
        else:
            if original_num_frames >= 2:
                seed_action_encoded = self.action_model.encode(video)
                seed_action_sequence = self.action_model.get_action_sequence(
                    seed_action_encoded
                ).long()
            else:
                seed_action_sequence = torch.empty(
                    video.shape[0], 0, dtype=torch.long, device=video.device
                )
        full_action_history = torch.cat(
            [seed_action_sequence, action_sequence], dim=1
        )

        generated = video
        for step in range(num_rollout_steps):
            # After ``step`` predictions we have ``S + step`` real frames.
            # The next code prediction uses the last ``context_size`` frames as
            # dynamics context (capped at ``T - 1``).
            current_num_frames = generated.shape[1]
            context_size = min(current_num_frames, max_context)
            context_start = current_num_frames - context_size
            context = generated[:, -context_size:]
            context_actions = full_action_history[
                :,
                context_start : current_num_frames - 1,
            ]
            next_codes = self.predict_next_frame_codes(
                context,
                action_sequence[:, step],
                max_steps=max_steps,
                context_actions=context_actions,
            )
            next_frame = self.decode_next_frame_with_tokenizer_window(
                generated,
                next_codes,
                decode_tokenizer=decode_tokenizer,
            )
            generated = torch.cat([generated, next_frame.unsqueeze(1)], dim=1)

        return generated

