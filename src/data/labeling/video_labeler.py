"""
Streamlit app for labeling video frames as valid or invalid.

Usage:
    pip install streamlit-image-coordinates
    streamlit run src/data/labeling/video_labeler.py

Click on frames to select spans:
- First click sets the START of selection
- Second click sets the END and marks the range as valid

Annotations are saved to: labeled_frames/{video_name}.json
"""

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False

# =============================================================================
# Configuration
# =============================================================================

FRAMES_PER_SECOND = 1  # Sample 1 frame per second of video
FRAMES_PER_WINDOW = 60  # Show 60 seconds (1 minute) at a time
THUMBNAIL_HEIGHT = 120  # Height of each thumbnail in the timeline
THUMBNAILS_PER_ROW = 15  # Number of thumbnails per row in grid
OUTPUT_DIR = Path("labeled_frames")
RAW_VIDEOS_DIR = Path("raw_videos")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# =============================================================================
# Annotation Management (path helpers first for use in discovery)
# =============================================================================


def get_output_path(video_path: str) -> Path:
    """Get the output JSON path for a video."""
    video_name = Path(video_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{video_name}.json"


# =============================================================================
# Video Discovery
# =============================================================================


def discover_videos(base_dir: Path = RAW_VIDEOS_DIR) -> list[Path]:
    """Discover all video files in base_dir and its subdirectories."""
    if not base_dir.exists():
        return []

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(base_dir.rglob(f"*{ext}"))

    return sorted(videos, key=lambda p: (p.parent.name, p.name))


def get_video_label_status(video_path: Path) -> tuple[bool, int]:
    """Check if a video has been labeled and how many intervals."""
    output_path = get_output_path(str(video_path))
    if output_path.exists():
        try:
            with open(output_path) as f:
                intervals = json.load(f)
                return True, len(intervals)
        except Exception:
            return False, 0
    return False, 0


def format_video_option(video_path: Path, base_dir: Path = RAW_VIDEOS_DIR) -> str:
    """Format a video path for display in the selectbox."""
    try:
        rel_path = video_path.relative_to(base_dir)
    except ValueError:
        rel_path = video_path

    is_labeled, num_intervals = get_video_label_status(video_path)

    if is_labeled:
        return f"✅ {rel_path} ({num_intervals} intervals)"
    else:
        return f"⬚ {rel_path}"


# =============================================================================
# Video Loading
# =============================================================================


@st.cache_data
def get_video_info(video_path: str) -> dict:
    """Get video metadata without loading all frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    cap.release()

    sample_interval = int(fps / FRAMES_PER_SECOND) if fps > 0 else 1
    sample_interval = max(1, sample_interval)
    sampled_frame_count = total_frames // sample_interval

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "sample_interval": sample_interval,
        "sampled_frame_count": sampled_frame_count,
    }


def load_frame_window(
    video_path: str,
    start_sample_idx: int,
    num_samples: int,
    sample_interval: int,
    thumb_height: int = THUMBNAIL_HEIGHT,
) -> list[tuple[int, Image.Image, int, int]]:
    """
    Load a window of sampled frames from the video.

    Returns list of (sample_index, thumbnail_image, thumb_width, thumb_height) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    for i in range(num_samples):
        sample_idx = start_sample_idx + i
        raw_frame_idx = sample_idx * sample_interval

        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        aspect = img.width / img.height
        thumb_width = int(thumb_height * aspect)
        img = img.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)

        frames.append((sample_idx, img, thumb_width, thumb_height))

    cap.release()
    return frames


# =============================================================================
# Annotation Management (continued)
# =============================================================================


def load_annotations(video_path: str) -> list[list[int]]:
    """Load existing annotations for a video."""
    output_path = get_output_path(video_path)
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return []


def save_annotations(video_path: str, intervals: list[list[int]]) -> None:
    """Save annotations to JSON."""
    output_path = get_output_path(video_path)
    with open(output_path, "w") as f:
        json.dump(intervals, f, indent=2)


def intervals_to_set(intervals: list[list[int]]) -> set[int]:
    """Convert list of [start, end] intervals to a set of frame indices."""
    result = set()
    for start, end in intervals:
        result.update(range(start, end + 1))
    return result


def set_to_intervals(frame_set: set[int]) -> list[list[int]]:
    """Convert a set of frame indices to sorted list of [start, end] intervals."""
    if not frame_set:
        return []

    sorted_frames = sorted(frame_set)
    intervals = []
    start = sorted_frames[0]
    end = sorted_frames[0]

    for frame in sorted_frames[1:]:
        if frame == end + 1:
            end = frame
        else:
            intervals.append([start, end])
            start = frame
            end = frame

    intervals.append([start, end])
    return intervals


# =============================================================================
# UI Components
# =============================================================================


def format_time(seconds: int) -> str:
    """Format seconds as MM:SS."""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def create_timeline_grid(
    frames: list[tuple[int, Image.Image, int, int]],
    selected_indices: set[int],
    pending_start: int | None = None,
    hover_end: int | None = None,
    thumbnails_per_row: int = THUMBNAILS_PER_ROW,
) -> tuple[Image.Image, int, int, dict[int, tuple[int, int, int, int]]]:
    """
    Create a grid of frames with selection highlighting.

    Returns:
        - Grid image
        - Thumbnail width
        - Cell height (including label)
        - Dict mapping frame index to (x, y, width, height) bounding box
    """
    if not frames:
        return Image.new("RGB", (400, 100), color=(50, 50, 50)), 0, 0, {}

    # Get thumbnail dimensions from first frame
    _, first_thumb, thumb_w, thumb_h = frames[0]

    # Add space for time label on each thumbnail
    label_height = 20
    cell_height = thumb_h + label_height

    # Calculate grid dimensions
    num_frames = len(frames)
    num_rows = (num_frames + thumbnails_per_row - 1) // thumbnails_per_row
    grid_w = thumbnails_per_row * thumb_w
    grid_h = num_rows * cell_height

    # Create grid image
    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)

    # Track frame bounding boxes for click detection
    frame_boxes: dict[int, tuple[int, int, int, int]] = {}

    for i, (sample_idx, thumb, tw, th) in enumerate(frames):
        row = i // thumbnails_per_row
        col = i % thumbnails_per_row
        x = col * thumb_w
        y = row * cell_height

        # Store bounding box for this frame
        frame_boxes[sample_idx] = (x, y + label_height, x + thumb_w, y + cell_height)

        thumb_arr = np.array(thumb)

        # Determine selection state
        is_selected = sample_idx in selected_indices

        # Check if in pending selection range
        in_pending_range = False
        if pending_start is not None and hover_end is not None:
            range_min = min(pending_start, hover_end)
            range_max = max(pending_start, hover_end)
            in_pending_range = range_min <= sample_idx <= range_max
        elif pending_start is not None and pending_start == sample_idx:
            in_pending_range = True

        # Apply highlighting
        if is_selected:
            # Already marked valid - green tint
            thumb_arr[:, :, 1] = np.clip(thumb_arr[:, :, 1] + 50, 0, 255)
            border_color = (0, 220, 0)
            label_color = (0, 220, 0)
        elif in_pending_range:
            # In pending selection - yellow/orange tint
            thumb_arr[:, :, 0] = np.clip(thumb_arr[:, :, 0] + 40, 0, 255)
            thumb_arr[:, :, 1] = np.clip(thumb_arr[:, :, 1] + 30, 0, 255)
            border_color = (255, 180, 0)
            label_color = (255, 180, 0)
        else:
            border_color = None
            label_color = (100, 100, 100)

        # Add border if selected
        if border_color:
            thumb_arr[:4, :, :] = border_color
            thumb_arr[-4:, :, :] = border_color
            thumb_arr[:, :4, :] = border_color
            thumb_arr[:, -4:, :] = border_color

        thumb_modified = Image.fromarray(thumb_arr)
        grid.paste(thumb_modified, (x, y + label_height))

        # Draw time label above the thumbnail
        time_str = format_time(sample_idx)
        draw.text((x + 4, y + 2), time_str, fill=label_color)

    return grid, thumb_w, cell_height, frame_boxes


def click_to_frame_index(
    click_x: int,
    click_y: int,
    thumb_w: int,
    cell_height: int,
    window_start: int,
    total_frames_in_window: int,
    thumbnails_per_row: int = THUMBNAILS_PER_ROW,
) -> int | None:
    """Convert click coordinates to frame index."""
    label_height = 20

    col = click_x // thumb_w
    row = click_y // cell_height

    # Check if click is in label area (above thumbnails in the row)
    y_in_cell = click_y % cell_height
    if y_in_cell < label_height:
        return None  # Clicked on label, not frame

    frame_offset = row * thumbnails_per_row + col

    if frame_offset < 0 or frame_offset >= total_frames_in_window:
        return None

    return window_start + frame_offset


# =============================================================================
# Main App
# =============================================================================


def main():
    st.set_page_config(
        page_title="Video Frame Labeler",
        page_icon="🎮",
        layout="wide",
    )

    # Check for required package
    if not HAS_IMAGE_COORDINATES:
        st.error(
            "⚠️ Missing required package. Please install it:\n\n"
            "```\npip install streamlit-image-coordinates\n```"
        )
        return

    st.title("🎮 Pokemon Video Frame Labeler")

    # Instructions
    st.markdown(
        """
        **Click on frames to select valid spans:**
        1. 🟡 **First click** → Sets START of selection (yellow highlight)
        2. 🟢 **Second click** → Sets END and marks range as VALID (green)
        """
    )

    # Sidebar: Video selection
    st.sidebar.header("📁 Video Selection")

    videos = discover_videos()

    if not videos:
        st.warning(
            f"No videos found in `{RAW_VIDEOS_DIR}/`. "
            "Please add video files to that directory."
        )
        return

    video_options = {format_video_option(v): v for v in videos}

    labeled_count = sum(1 for v in videos if get_video_label_status(v)[0])
    st.sidebar.markdown(f"**{labeled_count}/{len(videos)}** videos labeled")

    filter_option = st.sidebar.radio(
        "Show",
        ["All", "Unlabeled only", "Labeled only"],
        horizontal=True,
    )

    if filter_option == "Unlabeled only":
        video_options = {k: v for k, v in video_options.items() if k.startswith("⬚")}
    elif filter_option == "Labeled only":
        video_options = {k: v for k, v in video_options.items() if k.startswith("✅")}

    if not video_options:
        st.info("No videos match the current filter.")
        return

    selected_label = st.sidebar.selectbox(
        "Select video",
        options=list(video_options.keys()),
    )

    video_path_obj = video_options[selected_label]
    video_path = str(video_path_obj)

    # Load video info
    try:
        video_info = get_video_info(video_path)
    except Exception as e:
        st.error(f"Error loading video: {e}")
        return

    # Display video info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Video Info")
    duration_str = format_time(int(video_info["duration_seconds"]))
    st.sidebar.text(f"Duration: {duration_str}")
    st.sidebar.text(f"Total seconds: {video_info['sampled_frame_count']}")

    # Initialize session state
    if "selected_frames" not in st.session_state:
        st.session_state.selected_frames = set()

    if "loaded_video" not in st.session_state:
        st.session_state.loaded_video = None

    if "window_position" not in st.session_state:
        st.session_state.window_position = 0

    if "pending_start" not in st.session_state:
        st.session_state.pending_start = None

    if "processed_clicks" not in st.session_state:
        st.session_state.processed_clicks = set()

    # Load existing annotations if switching videos
    if st.session_state.loaded_video != video_path:
        existing = load_annotations(video_path)
        st.session_state.selected_frames = intervals_to_set(existing)
        st.session_state.loaded_video = video_path
        st.session_state.window_position = 0
        st.session_state.pending_start = None
        st.session_state.processed_clicks = set()

    total_samples = video_info["sampled_frame_count"]
    max_window_start = max(0, total_samples - FRAMES_PER_WINDOW)

    # Navigation controls
    st.markdown("---")

    nav_cols = st.columns([1, 1, 4, 1, 1])

    with nav_cols[0]:
        if st.button("⏮️ Start", use_container_width=True):
            st.session_state.window_position = 0
            st.rerun()

    with nav_cols[1]:
        if st.button("◀ Back", use_container_width=True):
            st.session_state.window_position = max(
                0, st.session_state.window_position - FRAMES_PER_WINDOW
            )
            st.rerun()

    with nav_cols[2]:
        window_start = st.slider(
            "Timeline position",
            min_value=0,
            max_value=max(1, max_window_start),
            value=st.session_state.window_position,
            step=FRAMES_PER_WINDOW // 4,
            format="%d sec",
            label_visibility="collapsed",
        )
        if window_start != st.session_state.window_position:
            st.session_state.window_position = window_start

    with nav_cols[3]:
        if st.button("▶ Next", use_container_width=True):
            st.session_state.window_position = min(
                max_window_start, st.session_state.window_position + FRAMES_PER_WINDOW
            )
            st.rerun()

    with nav_cols[4]:
        if st.button("⏭️ End", use_container_width=True):
            st.session_state.window_position = max_window_start
            st.rerun()

    # Current window info
    window_start = st.session_state.window_position
    window_end = min(window_start + FRAMES_PER_WINDOW, total_samples)
    current_window_size = window_end - window_start

    st.caption(
        f"📍 **{format_time(window_start)}** to **{format_time(window_end - 1)}** "
        f"({current_window_size} frames)"
    )

    # Load frames
    with st.spinner("Loading frames..."):
        frames = load_frame_window(
            video_path,
            start_sample_idx=window_start,
            num_samples=current_window_size,
            sample_interval=video_info["sample_interval"],
        )

    if not frames:
        st.warning("No frames loaded for this window.")
        return

    # Show pending selection status
    if st.session_state.pending_start is not None:
        st.info(
            f"🟡 Selection started at **{format_time(st.session_state.pending_start)}** — "
            f"Click another frame to complete the selection"
        )
        if st.button("❌ Cancel Selection"):
            st.session_state.pending_start = None
            st.rerun()

    # Create timeline grid
    timeline_img, thumb_w, cell_height, frame_boxes = create_timeline_grid(
        frames,
        st.session_state.selected_frames,
        pending_start=st.session_state.pending_start,
        hover_end=None,  # Could add hover tracking with more complex JS
    )

    # Display clickable timeline
    st.markdown("### 📽️ Click on frames to select")

    coords = streamlit_image_coordinates(
        timeline_img,
        key=f"timeline_{video_path}_{window_start}",
    )

    # Handle click - only process NEW clicks (not repeated from rerun)
    if coords is not None:
        click_x = coords["x"]
        click_y = coords["y"]

        # Create a unique identifier for this click
        click_id = (click_x, click_y, window_start)

        # Only process if this click hasn't been processed before
        if click_id not in st.session_state.processed_clicks:
            st.session_state.processed_clicks.add(click_id)

            # Limit set size to prevent unbounded growth
            if len(st.session_state.processed_clicks) > 100:
                st.session_state.processed_clicks = {click_id}

            # Scale coordinates if image was resized
            img_width, img_height = timeline_img.size
            if "width" in coords and coords["width"] > 0:
                scale_x = img_width / coords["width"]
                scale_y = img_height / coords["height"]
                click_x = int(click_x * scale_x)
                click_y = int(click_y * scale_y)

            clicked_frame = click_to_frame_index(
                click_x,
                click_y,
                thumb_w,
                cell_height,
                window_start,
                current_window_size,
            )

            if clicked_frame is not None:
                if st.session_state.pending_start is None:
                    # First click - set start
                    st.session_state.pending_start = clicked_frame
                    st.rerun()
                else:
                    # Second click - complete selection
                    start_frame = st.session_state.pending_start
                    end_frame = clicked_frame

                    # Ensure proper order
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame

                    # Add all frames in range to selected
                    for i in range(start_frame, end_frame + 1):
                        st.session_state.selected_frames.add(i)

                    st.session_state.pending_start = None
                    st.rerun()

    # Legend
    legend_cols = st.columns(3)
    with legend_cols[0]:
        st.markdown("🟢 **Green** = Marked valid")
    with legend_cols[1]:
        st.markdown("🟡 **Yellow** = Pending selection")
    with legend_cols[2]:
        st.markdown("⬛ **Dark** = Not selected")

    # Quick actions
    st.markdown("---")
    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("✅ Mark ALL in window", use_container_width=True):
            for i in range(window_start, window_end):
                st.session_state.selected_frames.add(i)
            st.rerun()

    with action_cols[1]:
        if st.button("❌ Unmark ALL in window", use_container_width=True):
            for i in range(window_start, window_end):
                st.session_state.selected_frames.discard(i)
            st.rerun()

    with action_cols[2]:
        if st.button("🔄 Toggle window", use_container_width=True):
            for i in range(window_start, window_end):
                if i in st.session_state.selected_frames:
                    st.session_state.selected_frames.discard(i)
                else:
                    st.session_state.selected_frames.add(i)
            st.rerun()

    with action_cols[3]:
        if st.button("🗑️ Clear ALL", use_container_width=True):
            st.session_state.selected_frames = set()
            st.session_state.pending_start = None
            st.rerun()

    # Summary and save
    st.markdown("---")

    intervals = set_to_intervals(st.session_state.selected_frames)
    total_selected = len(st.session_state.selected_frames)

    summary_cols = st.columns([2, 2, 2])

    with summary_cols[0]:
        st.metric("Selected Frames", f"{total_selected:,} sec")

    with summary_cols[1]:
        st.metric("Intervals", len(intervals))

    with summary_cols[2]:
        coverage = (total_selected / total_samples * 100) if total_samples > 0 else 0
        st.metric("Coverage", f"{coverage:.1f}%")

    # Save section
    save_cols = st.columns([1, 2])

    with save_cols[0]:
        if st.button("💾 Save Annotations", type="primary", use_container_width=True):
            save_annotations(video_path, intervals)
            st.success(f"✅ Saved {len(intervals)} intervals!")
            st.rerun()

    with save_cols[1]:
        output_path = get_output_path(video_path)
        st.caption(f"Output: `{output_path}`")

    # Intervals summary
    if intervals:
        with st.expander(f"📋 View {len(intervals)} intervals"):
            for i, (start, end) in enumerate(intervals[:20]):
                st.text(
                    f"{i + 1}. [{format_time(start)} - {format_time(end)}] "
                    f"({end - start + 1} sec)"
                )
            if len(intervals) > 20:
                st.text(f"... and {len(intervals) - 20} more")


if __name__ == "__main__":
    main()
