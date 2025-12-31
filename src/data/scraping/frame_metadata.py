from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FrameMetadata:
    """Metadata for extracted frames."""

    video_id: str
    frame_number: int
    timestamp: float
    game: str
    original_resolution: Tuple[int, int]
    cropped_resolution: Tuple[int, int]
    final_resolution: Tuple[int, int]

    # This is the path of the previous valid frame.
    prev_frame_key: Optional[str]
