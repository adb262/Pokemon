import logging
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader

from data.data_loaders.deterministic_shuffle_sampler import DeterministicShuffleSampler

logger = logging.getLogger(__name__)


class ResumableDataLoader:
    """Thin wrapper that pairs a ``DataLoader`` with a deterministic shuffle
    sampler to support O(1) mid-epoch resumes.

    Resuming is driven by the underlying ``DeterministicShuffleSampler``: the
    sampler is told which ``epoch`` permutation to use and how many leading
    batches to skip, so workers never load (and discard) the skipped data.
    This wrapper simply keeps ``current_epoch`` / ``current_batch`` state for
    checkpointing.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        sampler: Optional[DeterministicShuffleSampler] = None,
        start_epoch: int = 0,
        start_batch: int = 0,
    ):
        self.dataloader = dataloader
        self.sampler = sampler
        self.current_epoch = start_epoch
        # ``current_batch`` is the absolute index of the next batch to be
        # yielded. After yielding batch ``k``, it advances to ``k + 1``.
        self.current_batch = start_batch

    def set_epoch(self, epoch: int) -> None:
        """Move to a new epoch; subsequent iteration starts at batch 0."""
        self.current_epoch = epoch
        self.current_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        """Skip ``start_batch`` batches at the start of the next iteration."""
        if start_batch < 0:
            raise ValueError(f"start_batch must be non-negative, got {start_batch}")
        self.current_batch = start_batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        # If the previous iteration ran to completion, ``current_batch``
        # points past the last batch of the epoch. Re-iterating from there
        # would tell the sampler to skip the entire epoch and silently yield
        # zero batches, which is what callers that don't manage epoch state
        # (e.g. ``eval_model``) see as ``avg_loss=inf`` on the second eval.
        # Auto-reset to a fresh pass in that case while still honoring
        # mid-epoch resume positions.
        num_batches = len(self.dataloader)
        if self.current_batch >= num_batches:
            self.current_batch = 0
        start_batch = self.current_batch
        if self.sampler is not None:
            self.sampler.set_epoch(self.current_epoch)
            self.sampler.set_start_batch(start_batch)
        for batch in self.dataloader:
            yield batch
            self.current_batch += 1

    def __len__(self) -> int:
        return len(self.dataloader)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing"""
        return {
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
        }
