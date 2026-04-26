"""Helpers for monitoring discrete action-code usage during training.

These complement :mod:`monitoring.codebook_usage`: where ``codebook_usage``
reports aggregate statistics (perplexity, entropy, unique-fraction), this module
exposes the most-frequently-used codes so they can be surfaced in compact log
lines.
"""

import torch


@torch.no_grad()
def get_top_code_counts(
    action_tokens: torch.Tensor, vocab_size: int, top_k: int = 10
) -> list[tuple[int, int]]:
    """Return the most frequently used action-code ids and their counts."""
    flat_tokens = action_tokens.reshape(-1).long().cpu()
    if flat_tokens.numel() == 0 or vocab_size <= 0:
        return []

    counts = torch.bincount(flat_tokens, minlength=vocab_size)
    top_counts, top_indices = torch.topk(counts, k=min(top_k, vocab_size))
    return [
        (int(code_idx), int(code_count))
        for code_idx, code_count in zip(top_indices.tolist(), top_counts.tolist())
        if code_count > 0
    ]


def format_top_code_counts(top_code_counts: list[tuple[int, int]]) -> str:
    """Format top code-count pairs for compact logging."""
    if not top_code_counts:
        return "none"
    return ", ".join(f"{code_idx}:{code_count}" for code_idx, code_count in top_code_counts)
