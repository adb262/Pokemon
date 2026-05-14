from einops.einops import rearrange
import torch
import torch.nn as nn
from torch.distributions import uniform

from latent_action_model.model import LatentActionVQVAE
from loss.loss_fns import (
    changed_patch_weighted_token_cross_entropy_loss,
    compute_target_residuals,
    next_frame_reconstruction_loss,
    next_frame_reconstruction_loss_l1,
    next_frame_reconstruction_residual_loss,
)
from torch_utilities.initialize import init_weights
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer
from video_tokenization.model import VideoTokenizer
import logging
import torch.nn.functional as F

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
    ):
        super(DynamicsModel, self).__init__()
        self.mask_ratio_lower_bound = mask_ratio_lower_bound
        self.mask_ratio_upper_bound = mask_ratio_upper_bound
        self.num_images_in_video = num_images_in_video
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_ratio_distribution = uniform.Uniform(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )

        self.tokenizer_vocab_size = tokenizer.get_vocab_size()
        self.action_vocab_size = action_model.action_vocab_size

        self.tokenizer = tokenizer
        self.action_model = action_model
        self._ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
        self.action_loss_fn = next_frame_reconstruction_residual_loss

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
    def cosine_scheduler(max_steps: int, current_step: int, device: torch.device):
        return 0.5 * (1 + torch.cos(torch.pi * torch.tensor(current_step) / torch.tensor(max_steps, device=device)))

    def forward(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # Return our predictions (batch_size, num_images_in_video, num_patches, vocab_size)
        # And the masked targets

        logger.debug(f"video shape: {video.shape}")
        # x is of shape (batch_size, num_images_in_video, num_patches)
        with torch.no_grad():
            # x is of shape (batch_size, num_images_in_video, num_patches)
            targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video)
            ).long()

        batch_size, _, num_patches = targets.shape
        mask_ratio = torch.empty(batch_size, 1, 1, device=targets.device).uniform_(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )
        mask = (
            torch.rand(batch_size, self.num_images_in_video - 1, num_patches, device=targets.device) < mask_ratio
        )  # Boolean mask over only the target frame

        # Save original targets for loss computation before masking
        original_targets = targets.clone()

        targets[:, 1:, :] = self.tokenizer.mask_codebook_tokens(
            targets[:, 1:, :].long(), mask
        )

        # TODO: Should this also use the mask?
        # (batch_size, num_images_in_video - 1, num_patches, d_model)
        action_video_encoded = self.action_model.encode(video)

        action_tokens = self.action_model.get_action_sequence(
            action_video_encoded
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

        reconstructed_action_video = self.action_model.decode(video, action_video_encoded)
        # video_residuals = compute_target_residuals(video)
        action_loss = self.action_loss_fn(video, reconstructed_action_video)

        # Only compute token loss on masked positions of the final frame.
        target_frame = original_targets[:, 1:, :].masked_fill(~mask, -100)

        # View both as 3d tensors for the loss function
        logger.debug(f"x shape: {x.shape}, targets shape: {targets.shape}")
        # predicted_tokens = rearrange(x[:, -1:, :, :], "b t p d -> b (t p) d")
        # target_tokens = rearrange(target_frame, "b t p -> b (t p)")
        # logger.debug(f"predicted_tokens shape: {predicted_tokens.shape}, target_tokens shape: {target_tokens.shape}")
        predicted_tokens = rearrange(x[:, 1:, :, :], "b t p d -> b (t p) d")
        target_tokens = rearrange(target_frame, "b t p -> b (t p)")
        # previous_frame_tokens = rearrange(
        #     original_targets[:, -2:-1, :], "b t p -> b (t p)"
        # )
        # current_frame_tokens = rearrange(
        #     original_targets[:, -1:, :], "b t p -> b (t p)"
        # ) python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 10 --save_dir dynamics_model_atari_pong_action_103m_5_frames_l1_ --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_l1 --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64
        # token_loss = changed_patch_weighted_token_cross_entropy_loss(
        #     predicted_tokens=predicted_tokens,
        #     target_tokens=target_tokens,
        #     previous_frame_tokens=previous_frame_tokens,
        #     current_frame_tokens=current_frame_tokens,
        #     changed_patch_loss_weight=self.changed_patch_loss_weight,
        # )
        token_loss = self._ce_loss_fn(
            predicted_tokens.transpose(1, 2), target_tokens
        )

        return x, token_loss, action_loss, action_tokens

    @torch.inference_mode()
    def _inference_window(
        self, video: torch.Tensor, action: torch.Tensor, max_steps: int = 25
    ) -> torch.Tensor:
        # video: (B, T, C, H, W). The last frame is treated as MASKED and
        # its tokens are iteratively filled in by MaskGIT. The first T-1
        # frames are the observed context.
        # Return: (B, T, C, H, W) pixel video. Frames 0..T-2 are the
        # tokenizer's reconstruction of the observed context; frame T-1 is
        # the prediction.
        assert video.shape[1] == self.num_images_in_video, (
            f"_inference_window expects {self.num_images_in_video} frames, "
            f"got {video.shape[1]}"
        )

        with torch.no_grad():
            # (B, T, P) codebook indices; last frame's entries will be masked below.
            targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video)
            ).long()

        batch_size, _, num_patches = targets.shape

        action_token = action.long().to(targets.device)
        if action_token.dim() == 1:
            action_token = action_token.unsqueeze(1)  # (B,) -> (B, 1)

        # Mask the last frame's tokens in place (the frame we will predict).
        targets[:, -1, :] = self.tokenizer.get_mask_token_idx()

        # Derive the T-2 observed actions from the first T-1 frames only;
        # we must not peek at the target frame. The final action slot is
        # filled by the user-supplied ``action`` (the transition into the
        # masked frame).
        action_video_encoded = self.action_model.encode(video[:, :-1])

        action_tokens = self.action_model.get_action_sequence(
            action_video_encoded
        )
        logger.debug(f"action_tokens shape: {action_tokens.shape}")
        logger.debug(f"action_token shape: {action_token.shape}")
        action_tokens = torch.cat([action_tokens.long(), action_token], dim=1)

        # (B, T-1, d_model): one action embedding per observed frame; added
        # to x[:, :-1] below so positions 0..T-2 carry the action taking
        # them to the next frame (the last of which is the masked target).
        action_embeddings = self.action_embedding(action_tokens.long())

        step = 0
        mask_locations = torch.full((batch_size, num_patches), True, dtype=torch.bool, device=targets.device)
        while step < max_steps:
            x = self.tokenizer_embedding(targets.long())
            logger.debug(f"action_embeddings shape: {action_embeddings.shape}")
            logger.debug(f"x shape: {x.shape}")

            # Unsqueeze to add to each patch in the sequence
            x[:, :-1, :] += action_embeddings.unsqueeze(2)

            x = self.decoder(x)

            # logits is of shape (batch_size, num_images_in_video, num_patches, vocab_size)
            logits = self.vocab_head(x)

            # Stop early if everything is filled
            if not mask_locations.any():
                break

            # MaskGIT schedule: γ(t/T) gives fraction that should REMAIN masked after step t
            # tokens_to_unmask = current_masked - target_remaining_masked
            ratio_remaining = self.cosine_scheduler(max_steps, step + 1, targets.device)
            target_still_masked = int(num_patches * ratio_remaining)
            current_masked = int(mask_locations.sum(dim=1).amax().item())
            tokens_to_update = max(0, current_masked - target_still_masked)

            # logits[:, -1] -> (batch_size, num_patches, vocab_size)
            probs = self.softmax(logits[:, -1])

            # Sample only masked positions to avoid zero-probability rows
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
                probs_masked = probs_masked / (probs_masked.sum(-1, keepdim=True) + 1e-9)
                sampled_masked = torch.multinomial(probs_masked, 1, replacement=False).squeeze(-1)
                samples_flat[mask_flat] = sampled_masked

            samples = samples_flat.view(batch_size, num_patches)

            # Score sampled tokens to pick which patches to commit this step
            sampled_scores = probs.gather(
                -1, samples.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            sampled_scores[samples < 0] = -1

            logger.debug(f"tokens_to_update: {tokens_to_update}")

            if tokens_to_update > 0:
                _, top_positions = torch.topk(
                    sampled_scores, tokens_to_update, dim=-1
                )
                top_tokens = samples.gather(1, top_positions)

                # Scatter sampled tokens into the target frame at chosen positions
                targets_last = targets[:, -1, :]
                targets_last = targets_last.scatter(1, top_positions, top_tokens)
                targets[:, -1, :] = targets_last

                # Mark those positions as filled
                mask_locations = mask_locations.scatter(
                    1, top_positions, torch.zeros_like(top_positions, dtype=torch.bool)
                )

            step += 1

        return self.tokenizer.decode_from_codes(targets.float())

    @torch.inference_mode()
    def inference(
        self, video: torch.Tensor, action: torch.Tensor, max_steps: int = 25
    ) -> torch.Tensor:
        """Predict the ``T``-th (last) frame of each ``T``-frame sliding window.

        For input of shape ``(B, N, C, H, W)`` with ``N >= T``:

        * Each sliding window ``video[:, k : k+T]`` predicts its own last
          frame (original index ``k + T - 1``) via MaskGIT, matching the
          training-time setup where the model predicts the final frame of a
          ``T``-frame clip.
        * The first ``N - T`` windows derive the transition into their
          target frame from the observed ``video`` (since the true next
          frame is present). The final window uses the user-provided
          ``action``.

        Returns ``(B, N, C, H, W)`` where frames ``0..T-2`` are copied from
        ``video`` and frames ``T-1..N-1`` are each the corresponding
        window's predicted last frame.
        """
        num_frames = video.shape[1]
        window_size = self.num_images_in_video
        if num_frames < window_size:
            raise ValueError(
                f"inference requires at least {window_size} frames, got {num_frames}"
            )

        batch_size = video.shape[0]
        num_windows = num_frames - window_size + 1

        # Stack sliding T-frame windows into the batch dimension.
        windows = torch.stack(
            [
                video[:, start : start + window_size]
                for start in range(num_windows)
            ],
            dim=1,
        )
        windows = windows.reshape(
            batch_size * num_windows, window_size, *video.shape[2:]
        )

        action_token = action.long().to(video.device)
        if action_token.dim() == 1:
            action_token = action_token.unsqueeze(1)  # (B,) -> (B, 1)

        if num_windows == 1:
            batched_actions = action_token.reshape(batch_size * num_windows)
        else:
            # Observed actions for windows 0..num_windows-2: the transition
            # into window k's target frame is action_{k+T-2 -> k+T-1}, i.e.
            # full_action_tokens[:, k + T - 2]. We encode the full video
            # once and slice the relevant range.
            full_action_encoded = self.action_model.encode(video)
            full_action_tokens = self.action_model.get_action_sequence(
                full_action_encoded
            )  # (B, N-1); index k == action from frame k to frame k+1.

            observed_action_tokens = full_action_tokens[
                :, window_size - 2 : window_size - 2 + (num_windows - 1)
            ]  # (B, num_windows - 1)

            batched_actions = torch.cat(
                [observed_action_tokens.long(), action_token], dim=1
            )  # (B, num_windows)
            batched_actions = batched_actions.reshape(batch_size * num_windows)

        decoded_windows = self._inference_window(
            windows, batched_actions, max_steps=max_steps
        )  # (B * num_windows, T, C, H, W)

        predicted_last = decoded_windows[:, -1].reshape(
            batch_size, num_windows, *video.shape[2:]
        )

        context = video[:, : window_size - 1]  # (B, T-1, C, H, W)
        return torch.cat([context, predicted_last], dim=1)

    @torch.inference_mode()
    def predict_next_frame(
        self, context: torch.Tensor, action: torch.Tensor, max_steps: int = 10
    ) -> torch.Tensor:
        """Predict the next frame after the observed context."""
        required_context = self.num_images_in_video - 1
        num_frames = context.shape[1]
        if num_frames < required_context:
            raise ValueError(
                "predict_next_frame requires at least "
                f"{required_context} context frames, got {num_frames}"
            )

        context_window = context[:, -required_context:]
        # ``_inference_window`` masks the final slot before decoding, so this
        # placeholder frame is only used to satisfy the tokenizer's fixed T.
        placeholder = context_window[:, -1:].clone()
        inference_window = torch.cat([context_window, placeholder], dim=1)
        decoded_window = self._inference_window(
            inference_window, action, max_steps=max_steps
        )
        return decoded_window[:, -1]

    @torch.inference_mode()
    def rollout(
        self, video: torch.Tensor, actions: torch.Tensor, max_steps: int = 10
    ) -> torch.Tensor:
        """Generate future frames autoregressively from an initial context."""
        required_context = self.num_images_in_video - 1
        num_frames = video.shape[1]
        if num_frames < required_context:
            raise ValueError(
                f"rollout requires at least {required_context} frames, got {num_frames}"
            )

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

        generated = video
        num_rollout_steps = action_sequence.shape[1]
        for step in range(num_rollout_steps):
            next_frame = self.predict_next_frame(
                generated, action_sequence[:, step], max_steps=max_steps
            )
            generated = torch.cat([generated, next_frame.unsqueeze(1)], dim=1)

        return generated

