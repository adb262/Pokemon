import torch
from torch import nn

from transformers.spatio_temporal_transformer import SpatioTemporalTransformer


class ActionMappingModel(nn.Module):
    def __init__(
        self,
        *,
        num_input_actions: int,
        num_output_actions: int,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
    ):
        super(ActionMappingModel, self).__init__()
        self.num_input_actions = num_input_actions
        self.num_output_actions = num_output_actions
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_sequence_length = max_sequence_length
        self.max_transition_length = max_sequence_length - 1
        if self.max_transition_length < 1:
            raise ValueError("max_sequence_length must be at least 2")

        self.mapping_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, num_output_actions),
        )
        self.action_embedding = nn.Embedding(num_input_actions, d_model)
        self.spatio_temporal_transformer = SpatioTemporalTransformer(
            num_images_in_video=self.max_transition_length,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )

    def forward(self, video_token_latents: torch.Tensor, actions_taken: torch.Tensor) -> torch.Tensor:
        """Predict latent-action codes from source-frame latents and real actions.

        ``video_token_latents`` has shape ``(B, T, P, D)`` and ``actions_taken``
        contains 0-8 joint Pong action IDs with shape ``(B, T-1)``. The output
        logits have shape ``(B, T-1, num_output_actions)``.
        """
        if actions_taken.dim() != 2:
            raise ValueError(
                "actions_taken must contain joint action IDs with shape (B, T), "
                f"got {tuple(actions_taken.shape)}"
            )
        video_token_latents = video_token_latents[:, -self.max_sequence_length:, :, :]
        actions_taken = actions_taken[:, -self.max_transition_length:].long()

        action_embeddings = self.action_embedding(actions_taken)
        video_token_latents = video_token_latents[:, :-1, :, :] + action_embeddings.unsqueeze(2)
        video_token_latents = self.spatio_temporal_transformer(video_token_latents)

        x = video_token_latents.mean(dim=2)
        x = self.mapping_head(x)
        return x

    def compute_loss(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        target_actions = target_actions[:, -self.max_transition_length:].long()
        if logits.shape[:2] != target_actions.shape:
            raise ValueError(
                "logits and target_actions must agree on batch/time dimensions: "
                f"{tuple(logits.shape[:2])} vs {tuple(target_actions.shape)}"
            )

        return self.loss_fn(
            logits.reshape(-1, self.num_output_actions),
            target_actions.reshape(-1).long(),
        )