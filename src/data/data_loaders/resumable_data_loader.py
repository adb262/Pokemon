import logging

# Add the parent directory to the path so we can import from idm
import sys
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ResumableDataLoader:
    """A wrapper around DataLoader that supports resumable training"""

    def __init__(
        self, dataloader: DataLoader, start_epoch: int = 0, start_batch: int = 0
    ):
        self.dataloader = dataloader
        self.start_epoch = start_epoch
        self.start_batch = start_batch
        self.current_epoch = start_epoch
        self.current_batch = 0

    def __iter__(self):
        """Iterate through the dataloader, skipping to the correct position if resuming"""
        self.current_batch = 0
        iterator = iter(self.dataloader)

        # If we're resuming from a specific batch, skip ahead
        if self.current_epoch == self.start_epoch and self.start_batch > 0:
            logger.info(
                f"Resuming from epoch {self.start_epoch}, batch {self.start_batch}"
            )

            # Skip batches
            for _ in range(self.start_batch):
                try:
                    next(iterator)
                    self.current_batch += 1
                except StopIteration:
                    break

        # Continue from where we left off
        for batch in iterator:
            yield batch
            self.current_batch += 1

    def __len__(self):
        return len(self.dataloader)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing"""
        return {
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "dataset_state": {},  # We'll handle this at the PokemonFrameLoader level
        }

    def set_epoch(self, epoch: int):
        """Set the current epoch"""
        self.current_epoch = epoch
        self.current_batch = 0
