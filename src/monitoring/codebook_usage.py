"""Codebook usage metrics for quantized action tokens.

Given a batch of discrete action indices drawn from a codebook of size
``vocab_size``, these helpers report how uniformly the codebook is being
exercised. A healthy codebook should have high utilization (many unique codes)
and high perplexity (close to the vocab size).
"""

import math

import torch


@torch.no_grad()
def compute_codebook_usage(
    action_tokens: torch.Tensor,
    vocab_size: int,
) -> dict[str, float]:
    """Summarize how a batch of action indices uses a codebook.

    Args:
        action_tokens: Integer tensor of any shape containing indices in
            ``[0, vocab_size)``.
        vocab_size: Total size of the codebook.

    Returns:
        Dict with:
            usage_fraction: unique codes used / vocab_size (in [0, 1]).
            entropy_bits: entropy of the empirical code distribution, in bits.
            normalized_entropy: entropy_bits / log2(vocab_size), in [0, 1].
            perplexity: 2**entropy_bits, the effective number of codes used.
            num_unique: count of distinct codes observed.
            num_tokens: total number of tokens considered.
    """
    flat = action_tokens.reshape(-1).long()
    num_tokens = flat.numel()

    if num_tokens == 0 or vocab_size <= 0:
        return {
            "usage_fraction": 0.0,
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "perplexity": 0.0,
            "num_unique": 0.0,
            "num_tokens": 0.0,
        }

    counts = torch.bincount(flat, minlength=vocab_size).float()
    probs = counts / counts.sum()

    nonzero = probs > 0
    entropy_nats = -(probs[nonzero] * probs[nonzero].log()).sum().item()
    entropy_bits = entropy_nats / math.log(2)
    perplexity = math.exp(entropy_nats)

    num_unique = int(nonzero.sum().item())
    usage_fraction = num_unique / vocab_size
    max_entropy_bits = math.log2(vocab_size) if vocab_size > 1 else 1.0
    normalized_entropy = entropy_bits / max_entropy_bits if max_entropy_bits > 0 else 0.0

    return {
        "usage_fraction": usage_fraction,
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized_entropy,
        "perplexity": perplexity,
        "num_unique": float(num_unique),
        "num_tokens": float(num_tokens),
    }
