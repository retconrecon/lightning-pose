"""SAM3→SAM2 integration utilities for multi-animal pose estimation.

Converts SAM segmentation masks to Lightning Pose bbox format,
and provides pipelines for preparing per-animal cropped datasets
and running multi-animal inference.

Architecture:
    SAM3 detects animals via text prompt on frame 1.
    SAM2-tiny propagates tracking across all frames.
    Masks are converted to LP-compatible bboxes for cropping.
"""

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from typeguard import typechecked

from lightning_pose.utils import cropzoom, io

logger = logging.getLogger(__name__)

__all__ = [
    "sam_masks_to_lp_bbox",
    "lp_keypoints_to_sam_bbox",
    "smooth_bbox",
    "normalize_variance",
    "detect_swap_events",
    "apply_mask_to_images",
    "crop_frames_for_labeling",
    "prepare_animal_dataset",
    "prepare_all_animal_datasets",
    "run_multi_animal_inference",
    "merge_multi_animal_predictions",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_animal_id(raw_id: str) -> str:
    """Sanitize an animal ID for safe use in filesystem paths.

    Strips anything except alphanumeric characters, underscores, and hyphens.
    Logs a warning if the ID was modified.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_id)
    if sanitized != raw_id:
        logger.warning(f"Sanitized animal ID: {raw_id!r} -> {sanitized!r}")
    if not sanitized:
        raise ValueError(f"Animal ID is empty after sanitization: {raw_id!r}")
    return sanitized


def _write_provenance(output_dir: Path, extra: dict | None = None) -> Path:
    """Write provenance metadata to output directory.

    Records LP version, Python/system info, timestamp, and any extra
    config params so outputs are traceable back to the run that produced them.
    """
    import lightning_pose

    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "cpu"
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except ImportError:
        torch_version = cuda_version = cudnn_version = "unknown"

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lightning_pose_version": getattr(lightning_pose, "__version__", "unknown"),
        "python_version": sys.version,
        "command": " ".join(sys.argv),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "numpy_version": np.__version__,
    }
    if extra:
        meta.update(extra)

    provenance_file = output_dir / "provenance.json"
    provenance_file.write_text(json.dumps(meta, indent=2))
    logger.info(f"Provenance metadata written to {provenance_file}")
    return provenance_file


def _validate_bbox_df(bbox_df: pd.DataFrame, max_dimension: int = 8192) -> None:
    """Validate bbox DataFrame at stage boundaries.

    Checks for Inf values, non-positive dimensions, and extreme coordinates
    on non-NaN rows. NaN rows are allowed (they indicate empty masks /
    untracked frames).

    Args:
        bbox_df: DataFrame with columns [x, y, h, w].
        max_dimension: Maximum allowed bbox dimension (default 8192 = 8K).
            Prevents OOM from crafted/buggy bbox values.

    """
    non_nan_rows = bbox_df.dropna()
    if len(non_nan_rows) == 0:
        return  # All NaN — caller must handle this case
    if np.isinf(non_nan_rows.values).any():
        raise ValueError("Bbox DataFrame contains Inf values")
    if (non_nan_rows[["h", "w"]] <= 0).any().any():
        raise ValueError("Bbox DataFrame contains non-positive dimensions")
    if (non_nan_rows[["h", "w"]] > max_dimension).any().any():
        raise ValueError(
            f"Bbox dimensions exceed maximum {max_dimension}. "
            f"Max h={non_nan_rows['h'].max():.0f}, max w={non_nan_rows['w'].max():.0f}"
        )


# ---------------------------------------------------------------------------
# Mask → Bbox conversion
# ---------------------------------------------------------------------------


@typechecked
def sam_masks_to_lp_bbox(
    masks: np.ndarray,
    image_index: list | pd.Index,
    crop_ratio: float = 2.0,
) -> pd.DataFrame:
    """Convert SAM2 binary masks to LP-compatible bbox CSV format.

    Args:
        masks: Binary masks of shape (num_frames, H, W) for a single animal.
            Nonzero values indicate the animal's pixels.
        image_index: Image paths (relative to data_dir) or frame identifiers.
            Must have the same length as masks.shape[0].
        crop_ratio: Factor by which to expand the tight bounding box around the mask.
            1.0 = tight crop, 2.0 = 2x larger (recommended to include context).

    Returns:
        DataFrame with columns [x, y, h, w] and the given index.
        x, y = top-left corner; h, w = height, width of the bounding box.

    """
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with 3 dims (frames, H, W), got {masks.ndim}")
    if len(image_index) != masks.shape[0]:
        raise ValueError(
            f"image_index length ({len(image_index)}) != number of masks ({masks.shape[0]})"
        )
    if not np.isfinite(masks).all():
        raise ValueError("Mask array contains NaN or Inf values")
    if not (0.5 <= crop_ratio <= 10.0):
        raise ValueError(
            f"crop_ratio must be between 0.5 and 10.0, got {crop_ratio}"
        )

    frame_h, frame_w = masks.shape[1], masks.shape[2]
    bboxes = []
    empty_count = 0

    for frame_idx, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            # Empty mask — animal not detected in this frame.
            # Use NaN to propagate "missing" signal rather than producing
            # plausible-but-wrong centered bboxes.
            bboxes.append([np.nan, np.nan, np.nan, np.nan])
            empty_count += 1
            continue

        # Tight bbox around mask (+1 for inclusive pixel extent)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        tight_w = x_max - x_min + 1
        tight_h = y_max - y_min + 1

        # Use square bbox (max of w, h) scaled by crop_ratio
        bbox_size = max(tight_w, tight_h)
        bbox_size = int(np.ceil(bbox_size * crop_ratio))
        # Ensure even dimensions (video players prefer this)
        if bbox_size % 2 != 0:
            bbox_size += 1

        # Center bbox on mask centroid
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        x = int(cx - bbox_size / 2)
        y = int(cy - bbox_size / 2)

        bboxes.append([x, y, bbox_size, bbox_size])

    if empty_count > 0:
        pct = empty_count / len(masks) * 100
        logger.warning(
            f"{empty_count}/{len(masks)} frames ({pct:.1f}%) have empty masks. "
            f"These frames will have NaN bboxes."
        )
        if empty_count > len(masks) * 0.5:
            raise ValueError(
                f"More than 50% of frames ({empty_count}/{len(masks)}) have empty masks. "
                f"SAM tracking has likely failed for this animal."
            )

    return pd.DataFrame(bboxes, index=image_index, columns=["x", "y", "h", "w"])


@typechecked
def sam_masks_to_lp_bbox_batch(
    masks_per_animal: dict[str, np.ndarray],
    image_index: list | pd.Index,
    crop_ratio: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Convert masks for multiple animals into per-animal bbox DataFrames.

    Args:
        masks_per_animal: Dict mapping animal_id → mask array of shape (num_frames, H, W).
        image_index: Image paths or frame identifiers.
        crop_ratio: Factor to expand bounding boxes.

    Returns:
        Dict mapping animal_id → bbox DataFrame.

    """
    return {
        animal_id: sam_masks_to_lp_bbox(masks, image_index, crop_ratio=crop_ratio)
        for animal_id, masks in masks_per_animal.items()
    }


@typechecked
def lp_keypoints_to_sam_bbox(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    padding_ratio: float = 0.25,
    confidence_threshold: float = 0.3,
    min_confident_keypoints: int = 2,
    frame_dims: tuple[int, int] | None = None,
    min_bbox_size: int = 0,
) -> list[int] | None:
    """Convert LP keypoints to a SAM/SAMURAI-compatible [x1, y1, x2, y2] bbox.

    Filters keypoints by confidence, computes a tight bounding box around the
    confident keypoints, then pads by padding_ratio.

    This is the reverse direction of sam_masks_to_lp_bbox (which goes mask -> bbox).

    Bbox format reference::

        This function returns:    [x1, y1, x2, y2]  — corner coordinates
        predict_frame expects:    (x, y, w, h)       — top-left + size
        LP DataFrame columns:    [x, y, h, w]        — top-left + size (h before w)

    To convert this output to predict_frame input::

        result = lp_keypoints_to_sam_bbox(kp, conf)
        if result is not None:
            x1, y1, x2, y2 = result
            bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)

    Args:
        keypoints: (num_kp, 2) array of (x, y) coordinates in original frame space.
            NaN keypoints are filtered out before confidence thresholding.
        confidence: (num_kp,) array of per-keypoint confidence scores.
            Heatmap peak intensity (not a calibrated probability). The default
            threshold of 0.3 is a heuristic — optimal values depend on model
            and dataset.
        padding_ratio: Fraction of bbox size to add as padding on each side.
            0.25 means 25% padding, so the bbox is 1.5x the tight fit.
            Must be >= 0.
        confidence_threshold: Minimum confidence for a keypoint to be included.
        min_confident_keypoints: Minimum number of confident keypoints required
            to produce a bbox. Returns None if fewer pass the threshold.
        frame_dims: Optional (width, height) of the source frame. When provided,
            the output bbox is clamped to ``[0, 0, width, height]`` so coordinates
            never go negative or exceed frame boundaries.
        min_bbox_size: Minimum width and height (in pixels) of the output bbox.
            If the bbox is smaller than this in either dimension, it is expanded
            symmetrically around its center. Prevents tiny crops that amplify
            noise through ``cv2.resize``. Default 0 (no minimum).

    Returns:
        [x1, y1, x2, y2] integer bbox, or None if too few confident keypoints.

    """
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(
            f"keypoints must be (N, 2), got shape {keypoints.shape}"
        )
    if confidence.ndim != 1 or len(confidence) != len(keypoints):
        raise ValueError(
            f"confidence must be 1D with len matching keypoints "
            f"({len(keypoints)}), got shape {confidence.shape}"
        )
    if padding_ratio < 0:
        raise ValueError(f"padding_ratio must be >= 0, got {padding_ratio}")
    if min_bbox_size < 0:
        raise ValueError(f"min_bbox_size must be >= 0, got {min_bbox_size}")

    # Filter NaN keypoints before confidence check (AXIOM-5).
    # NaN coordinates would pass confidence filter and crash int().
    finite_mask = np.isfinite(keypoints).all(axis=1)
    conf_mask = confidence >= confidence_threshold
    mask = finite_mask & conf_mask
    if mask.sum() < min_confident_keypoints:
        return None

    kp = keypoints[mask]  # (M, 2)
    x_min, y_min = kp.min(axis=0)
    x_max, y_max = kp.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    x1 = int(x_min - pad_x)
    y1 = int(y_min - pad_y)
    x2 = int(x_max + pad_x)
    y2 = int(y_max + pad_y)

    # Enforce minimum bbox size by expanding symmetrically around center.
    if min_bbox_size > 0:
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_bbox_size:
            deficit = min_bbox_size - bw
            x1 -= deficit // 2
            x2 += (deficit + 1) // 2  # ceil for odd deficit
        if bh < min_bbox_size:
            deficit = min_bbox_size - bh
            y1 -= deficit // 2
            y2 += (deficit + 1) // 2

    # Clamp to frame boundaries.
    if frame_dims is not None:
        fw, fh = frame_dims
        if fw <= 0 or fh <= 0:
            raise ValueError(f"frame_dims must be positive, got {frame_dims}")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

    return [x1, y1, x2, y2]


@typechecked
def smooth_bbox(
    current: list[int],
    previous: list[int],
    alpha: float = 0.7,
) -> list[int]:
    """Exponential moving average of bbox coordinates across frames.

    Prevents jitter in the predict -> bbox -> crop -> predict feedback loop
    by smoothing bbox transitions. Higher ``alpha`` weights the current frame
    more heavily (less smoothing).

    Args:
        current: Current frame's ``[x1, y1, x2, y2]`` bbox.
        previous: Previous frame's ``[x1, y1, x2, y2]`` bbox.
        alpha: Weight for the current bbox. Must be in ``(0, 1]``.
            1.0 means no smoothing (return current as-is).
            0.5 means equal weight to current and previous.

    Returns:
        Smoothed ``[x1, y1, x2, y2]`` integer bbox.

    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if len(current) != 4 or len(previous) != 4:
        raise ValueError(
            f"current and previous must each have 4 elements, "
            f"got {len(current)} and {len(previous)}"
        )
    return [
        int(round(alpha * c + (1.0 - alpha) * p))
        for c, p in zip(current, previous)
    ]


@typechecked
def normalize_variance(
    variance: np.ndarray,
    calibration_stats: dict,
) -> np.ndarray:
    """Normalize per-frame variance to [0, 1] using calibration statistics.

    Makes variance thresholds portable across models and datasets by scaling
    raw variance (which depends on image resolution, model architecture, and
    ensemble size) to a consistent range.

    Uses percentile-based normalization: values at or below ``p5`` map to 0,
    values at or above ``p95`` map to 1, with linear interpolation between.
    Output is clipped to [0, 1].

    Collect calibration stats by running the ensemble on ~1000 "normal
    tracking" frames and recording ``np.percentile(variances, [5, 95])``.

    Args:
        variance: Per-keypoint or per-frame variance array (any shape).
            Units are px^2 from ``EnsemblePredictor.predict_frame``.
        calibration_stats: Must contain ``"p5"`` and ``"p95"`` keys with
            float values. ``p95`` must be strictly greater than ``p5``.

    Returns:
        Array of same shape as ``variance`` with values in [0, 1].

    """
    p5 = calibration_stats["p5"]
    p95 = calibration_stats["p95"]
    if not (np.isfinite(p5) and np.isfinite(p95)):
        raise ValueError(f"p5 and p95 must be finite, got p5={p5}, p95={p95}")
    if p95 <= p5:
        raise ValueError(f"p95 must be > p5, got p5={p5}, p95={p95}")
    if (p95 - p5) < 1e-6:
        raise ValueError(
            f"calibration range too narrow (p95 - p5 = {p95 - p5:.2e}), "
            f"variance distribution may be degenerate"
        )
    normalized = (variance - p5) / (p95 - p5)
    return np.clip(normalized, 0.0, 1.0)


@typechecked
def detect_swap_events(
    trajectories_a: np.ndarray,
    trajectories_b: np.ndarray,
    min_swap_fraction: float = 0.5,
    temporal_window: int = 5,
) -> list[dict[str, int | float]]:
    """Detect identity swap events from keypoint trajectory crossings.

    When two tracked animals cross paths, SAM2/SAMURAI often swaps their
    identities. This manifests as simultaneous sign flips in the relative
    position (A - B) across multiple bodyparts. A frame where a large fraction
    of bodyparts flip sign is likely a swap.

    Algorithm:
        1. Compute ``diff = trajectories_a - trajectories_b`` per frame.
        2. For each bodypart, detect frames where the sign of diff flips
           in either x or y (a "crossing").
        3. Flag frames where the fraction of crossing bodyparts >= threshold.
        4. Merge flagged frames within ``temporal_window`` into single events.

    Args:
        trajectories_a: (T, K, 2) keypoints for animal A across T frames,
            K bodyparts, 2 coordinates (x, y). NaN indicates missing data.
        trajectories_b: (T, K, 2) keypoints for animal B. Must match shape.
        min_swap_fraction: Fraction of bodyparts that must cross simultaneously
            to flag a swap. Must be in (0, 1].
        temporal_window: Consecutive flagged frames within this many frames
            of each other are merged into a single event. Must be >= 1.

    Returns:
        List of swap event dicts, each containing:
            - ``"frame"``: int — frame index (center of the crossing cluster)
            - ``"n_crossed"``: int — number of bodyparts that crossed
            - ``"n_valid"``: int — total non-NaN bodyparts at that frame
            - ``"fraction"``: float — n_crossed / n_valid

    """
    # --- Input validation ---
    if trajectories_a.ndim != 3 or trajectories_a.shape[2] != 2:
        raise ValueError(
            f"trajectories_a must be (T, K, 2), got {trajectories_a.shape}"
        )
    if trajectories_a.shape != trajectories_b.shape:
        raise ValueError(
            f"Shape mismatch: trajectories_a {trajectories_a.shape} "
            f"!= trajectories_b {trajectories_b.shape}"
        )
    if trajectories_a.shape[0] < 2:
        raise ValueError(
            f"Need >= 2 frames, got {trajectories_a.shape[0]}"
        )
    if not (0.0 < min_swap_fraction <= 1.0):
        raise ValueError(
            f"min_swap_fraction must be in (0, 1], got {min_swap_fraction}"
        )
    if temporal_window < 1:
        raise ValueError(
            f"temporal_window must be >= 1, got {temporal_window}"
        )

    T, K, _ = trajectories_a.shape

    # --- Compute sign of relative position ---
    diff = trajectories_a - trajectories_b  # (T, K, 2)

    # --- Detect per-bodypart crossings ---
    # A crossing at frame t means sign(diff[t]) != sign(diff[t-1]) for x or y.
    # np.sign maps negative→-1, zero→0, positive→+1.
    signs = np.sign(diff)  # (T, K, 2)

    # Identify valid (non-NaN) frames per bodypart: both animals must be finite
    valid_a = np.isfinite(trajectories_a).all(axis=2)  # (T, K)
    valid_b = np.isfinite(trajectories_b).all(axis=2)  # (T, K)
    valid = valid_a & valid_b  # (T, K)

    # Sign flip between consecutive frames for each coordinate
    sign_change = signs[1:] != signs[:-1]  # (T-1, K, 2)

    # A bodypart "crosses" if either x or y flips sign
    crossing_per_bp = sign_change.any(axis=2)  # (T-1, K)

    # Both current and previous frame must be valid for this bodypart
    both_valid = valid[1:] & valid[:-1]  # (T-1, K)

    # Only count crossings where both frames are valid
    crossing_per_bp = crossing_per_bp & both_valid  # (T-1, K)

    # --- Count crossings per frame and threshold ---
    flagged_frames = []  # list of (frame_index, n_crossed, n_valid, fraction)

    for t in range(T - 1):
        n_valid = both_valid[t].sum()
        if n_valid == 0:
            continue
        n_crossed = crossing_per_bp[t].sum()
        fraction = n_crossed / n_valid
        if fraction >= min_swap_fraction:
            # Report as frame t+1 (the frame where the new sign appears)
            flagged_frames.append((t + 1, int(n_crossed), int(n_valid), float(fraction)))

    if not flagged_frames:
        return []

    # --- Merge within temporal_window ---
    events: list[dict[str, int | float]] = []
    current_cluster: list[tuple[int, int, int, float]] = [flagged_frames[0]]

    for entry in flagged_frames[1:]:
        if entry[0] - current_cluster[-1][0] <= temporal_window:
            current_cluster.append(entry)
        else:
            # Emit event for completed cluster
            events.append(_cluster_to_event(current_cluster))
            current_cluster = [entry]

    # Emit final cluster
    events.append(_cluster_to_event(current_cluster))

    return events


def _cluster_to_event(
    cluster: list[tuple[int, int, int, float]],
) -> dict[str, int | float]:
    """Pick the frame with the highest fraction as the event center."""
    best = max(cluster, key=lambda x: x[3])
    return {
        "frame": best[0],
        "n_crossed": best[1],
        "n_valid": best[2],
        "fraction": best[3],
    }


# ---------------------------------------------------------------------------
# Mask-based background removal
# ---------------------------------------------------------------------------


@typechecked
def apply_mask_to_images(
    masks: np.ndarray,
    image_paths: list[str],
    root_directory: Path,
    output_directory: Path,
) -> None:
    """Zero out background pixels in images using SAM masks.

    This removes distractor animals so LP sees clean single-animal images.
    Uses thread-parallel I/O for faster processing on large datasets.

    Args:
        masks: Binary masks of shape (num_frames, H, W) for one animal.
        image_paths: Relative image paths (same order as masks).
        root_directory: Root directory containing the original images.
        output_directory: Where to write masked images.

    """

    def _process_single(idx_and_path: tuple[int, str]) -> None:
        i, img_rel_path = idx_and_path
        if ".." in Path(img_rel_path).parts:
            raise ValueError(f"Image path contains traversal: {img_rel_path}")
        img_path = root_directory / img_rel_path
        output_path = output_directory / img_rel_path

        img = np.array(Image.open(img_path).convert("RGB"))
        # Copy mask frame into thread-local memory to avoid mmap race conditions.
        # Memory-mapped arrays are not thread-safe for concurrent reads on all
        # platforms (NFS serializes page faults; macOS can invalidate mmap on GC).
        mask = np.array(masks[i])

        # Resize mask to image dimensions if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply mask (zero out background)
        mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
        masked_img = img * (mask_3d > 0).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(masked_img).save(output_path)

    import os as _os

    max_workers = min(8, _os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(tqdm.tqdm(
            pool.map(_process_single, enumerate(image_paths)),
            total=len(image_paths),
            desc="Applying masks",
        ))


# ---------------------------------------------------------------------------
# Crop frames for labeling (pre-annotation workflow)
# ---------------------------------------------------------------------------


@typechecked
def crop_frames_for_labeling(
    bbox_df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    expand: float = 1.0,
) -> Path:
    """Crop frames using SAM-derived bboxes for easier pose labeling.

    This is the entry point for the pre-annotation workflow: run SAM to get
    bounding boxes, then crop frames per animal so the labeler sees zoomed-in
    views. No labels CSV is needed — this runs before any annotation.

    The UI can call this with different ``expand`` values to let the user
    widen the crop if keypoints are getting cut off.

    Args:
        bbox_df: Bounding box DataFrame with columns [x, y, h, w].
            Typically loaded from the bbox CSV produced by ``sam_masks_to_lp_bbox``.
        input_dir: Directory containing the original full-frame images.
        output_dir: Where to write the cropped images.
        expand: Factor to expand bboxes beyond their current size.
            1.0 = no change, 1.5 = 50% larger, etc. Boxes stay centered
            and are clamped to frame boundaries.

    Returns:
        Path to the bbox CSV written alongside the cropped images
        (reflects the actual bboxes used after expansion).

    """
    if expand <= 0:
        raise ValueError(f"expand must be positive, got {expand}")

    if expand != 1.0:
        bbox_df = _expand_bboxes(bbox_df, expand)

    _validate_bbox_df(bbox_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_file = output_dir / "bbox.csv"
    bbox_df.to_csv(bbox_file)

    cropzoom._crop_images(bbox_df, input_dir, output_dir)
    logger.info(
        f"Cropped {len(bbox_df)} frames (expand={expand}) → {output_dir}"
    )

    return bbox_file


def _expand_bboxes(bbox_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale bboxes by a factor while keeping them centered.

    Args:
        bbox_df: DataFrame with columns [x, y, h, w].
        factor: Expansion multiplier (>1 = larger, <1 = smaller).

    Returns:
        New DataFrame with expanded bboxes. NaN rows are preserved.

    """
    df = bbox_df.copy()
    valid = df.dropna().index

    # Current center
    cx = df.loc[valid, "x"] + df.loc[valid, "w"] / 2
    cy = df.loc[valid, "y"] + df.loc[valid, "h"] / 2

    # Scale dimensions
    new_h = df.loc[valid, "h"] * factor
    new_w = df.loc[valid, "w"] * factor

    # Ensure even dimensions
    new_h = (np.ceil(new_h / 2) * 2).astype(int)
    new_w = (np.ceil(new_w / 2) * 2).astype(int)

    # Recenter
    df.loc[valid, "x"] = (cx - new_w / 2).astype(int)
    df.loc[valid, "y"] = (cy - new_h / 2).astype(int)
    df.loc[valid, "h"] = new_h
    df.loc[valid, "w"] = new_w

    return df


# ---------------------------------------------------------------------------
# Per-animal dataset preparation
# ---------------------------------------------------------------------------


@typechecked
def prepare_animal_dataset(
    animal_id: str,
    bbox_df: pd.DataFrame,
    input_data_dir: Path,
    input_csv_file: Path,
    output_base_dir: Path,
    masks: np.ndarray | None = None,
    image_paths: list[str] | None = None,
) -> dict[str, Path]:
    """Prepare a cropped dataset for a single animal.

    Uses LP's existing crop utilities to create a per-animal dataset
    with cropped images and adjusted CSV coordinates.

    Args:
        animal_id: Identifier for the animal (e.g., "animal_0").
        bbox_df: Bounding box DataFrame with columns [x, y, h, w].
        input_data_dir: Path to the original data directory with full-frame images.
        input_csv_file: Path to the original labels CSV.
        output_base_dir: Base directory for all animal outputs.
        masks: Optional binary masks (num_frames, H, W) for background removal.
            If provided, background pixels are zeroed before cropping.
        image_paths: Required if masks is provided. Relative image paths.

    Returns:
        Dict with paths to output files:
            - "data_dir": cropped images directory
            - "csv_file": adjusted labels CSV
            - "bbox_file": bbox CSV

    """
    animal_id = _sanitize_animal_id(animal_id)
    animal_dir = output_base_dir / animal_id
    cropped_data_dir = animal_dir / "cropped_data"
    bbox_file = animal_dir / "bbox.csv"
    cropped_csv_file = animal_dir / "CollectedData_cropped.csv"

    # Save bbox CSV
    animal_dir.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(bbox_file)

    # Optionally apply mask-based background removal before cropping
    if masks is not None:
        if image_paths is None:
            raise ValueError("image_paths required when masks are provided")
        masked_data_dir = animal_dir / "masked_data"
        apply_mask_to_images(masks, image_paths, input_data_dir, masked_data_dir)
        crop_source_dir = masked_data_dir
    else:
        crop_source_dir = input_data_dir

    # Crop images using LP's existing utilities
    cropzoom._crop_images(bbox_df, crop_source_dir, cropped_data_dir)

    # Adjust CSV coordinates (subtract bbox offsets)
    cropzoom.generate_cropped_csv_file(
        input_csv_file=input_csv_file,
        input_bbox_file=bbox_file,
        output_csv_file=cropped_csv_file,
        mode="subtract",
    )

    return {
        "data_dir": cropped_data_dir,
        "csv_file": cropped_csv_file,
        "bbox_file": bbox_file,
    }


@typechecked
def prepare_all_animal_datasets(
    bbox_dfs: dict[str, pd.DataFrame],
    input_data_dir: Path,
    input_csv_file: Path,
    output_base_dir: Path,
    masks_per_animal: dict[str, np.ndarray] | None = None,
    image_paths: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Prepare cropped datasets for all detected animals.

    Args:
        bbox_dfs: Dict mapping animal_id → bbox DataFrame.
        input_data_dir: Path to original data directory.
        input_csv_file: Path to original labels CSV.
        output_base_dir: Base output directory.
        masks_per_animal: Optional dict mapping animal_id → masks array.
        image_paths: Relative image paths (required if masks provided).

    Returns:
        Dict mapping animal_id → dict of output paths.

    """
    results = {}
    for animal_id, bbox_df in bbox_dfs.items():
        animal_masks = masks_per_animal.get(animal_id) if masks_per_animal else None
        results[animal_id] = prepare_animal_dataset(
            animal_id=animal_id,
            bbox_df=bbox_df,
            input_data_dir=input_data_dir,
            input_csv_file=input_csv_file,
            output_base_dir=output_base_dir,
            masks=animal_masks,
            image_paths=image_paths,
        )
    return results


# ---------------------------------------------------------------------------
# Multi-animal inference
# ---------------------------------------------------------------------------


@typechecked
def crop_video_for_animal(
    animal_id: str,
    bbox_df: pd.DataFrame,
    input_video_file: Path,
    output_base_dir: Path,
) -> dict[str, Path]:
    """Crop a video for a single animal using SAM-derived bboxes.

    Args:
        animal_id: Animal identifier.
        bbox_df: Per-frame bounding boxes for this animal.
        input_video_file: Path to the original video.
        output_base_dir: Base directory for outputs.

    Returns:
        Dict with paths:
            - "cropped_video": path to cropped video
            - "bbox_file": path to saved bbox CSV

    """
    animal_id = _sanitize_animal_id(animal_id)
    _validate_bbox_df(bbox_df)

    animal_dir = output_base_dir / animal_id
    animal_dir.mkdir(parents=True, exist_ok=True)
    bbox_file = animal_dir / "bbox.csv"
    cropped_video = animal_dir / f"cropped_{input_video_file.name}"

    # Validate bbox row count matches video frame count.
    # A mismatch usually means masks were generated from a different video.
    # Allow tolerance of 5 frames for codec frame-count rounding.
    cap = cv2.VideoCapture(str(input_video_file))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if abs(len(bbox_df) - n_video_frames) > 5:
        raise ValueError(
            f"Animal {animal_id}: bbox DataFrame has {len(bbox_df)} rows "
            f"but video has {n_video_frames} frames (diff={abs(len(bbox_df) - n_video_frames)}). "
            f"Ensure masks/bboxes were generated from the same video."
        )

    bbox_df.to_csv(bbox_file)

    # Fill NaN bboxes with nearest valid frame's bbox for video cropping.
    # NaN frames (empty masks / tracking loss) would crash MoviePy because
    # int(NaN) raises ValueError. Forward-fill then backward-fill uses the
    # nearest known position. The original NaN markers are preserved in
    # bbox_df (saved to CSV above) so remap_predictions can re-introduce
    # NaN for these frames downstream.
    bbox_df_filled = bbox_df.ffill().bfill()
    if bbox_df_filled.isna().any().any():
        raise ValueError(
            f"Animal {animal_id}: cannot fill NaN bboxes — all frames are empty. "
            f"This animal has no valid tracking data."
        )
    cropzoom._crop_video_moviepy(input_video_file, bbox_df_filled, cropped_video)

    return {
        "cropped_video": cropped_video,
        "bbox_file": bbox_file,
    }


@typechecked
def remap_predictions(
    preds_file: Path,
    bbox_file: Path,
    output_file: Path | None = None,
) -> Path:
    """Remap predictions from cropped coordinates back to original frame.

    Args:
        preds_file: Path to predictions CSV (in cropped coordinates).
        bbox_file: Path to the bbox CSV used for cropping.
        output_file: Where to save remapped predictions.
            Defaults to remapped_<preds_file.name> in the same directory.

    Returns:
        Path to the remapped predictions file.

    """
    if output_file is None:
        output_file = preds_file.with_name(f"remapped_{preds_file.name}")

    cropzoom.generate_cropped_csv_file(
        input_csv_file=preds_file,
        input_bbox_file=bbox_file,
        output_csv_file=output_file,
        mode="add",
    )

    # Validate remap didn't silently produce all-NaN due to index type mismatch.
    # pandas aligns Series by index during addition; if prediction index (e.g.
    # string "0") doesn't match bbox index (e.g. integer 0), every row becomes NaN.
    result_df = pd.read_csv(output_file, header=[0, 1, 2], index_col=0)
    xy_cols = [c for c in result_df.columns if c[-1] in ("x", "y")]
    if len(xy_cols) > 0 and result_df[xy_cols].isna().all().all():
        raise RuntimeError(
            f"Remap produced all-NaN coordinates. This usually means the "
            f"prediction index and bbox index have different types "
            f"(string vs integer). Check {preds_file} and {bbox_file}."
        )

    return output_file



@typechecked
def run_multi_animal_inference(
    bbox_dfs: dict[str, pd.DataFrame],
    input_video_file: Path,
    model_dir: Path | list[Path],
    output_base_dir: Path,
    compute_metrics: bool = False,
) -> dict[str, dict[str, Path]]:
    """Run LP inference for each detected animal in a video.

    For each animal:
    1. Crop the video using SAM-derived bboxes (once, regardless of model count)
    2. Run LP prediction on the cropped video
    3. Remap predictions to original frame coordinates

    When ``model_dir`` is a list of paths (ensemble mode), each model runs
    inference on the same cropped video.  Per-animal ensemble mean and variance
    are computed via :func:`compute_ensemble_variance` and saved alongside the
    individual model predictions.

    Args:
        bbox_dfs: Dict mapping animal_id → per-frame bbox DataFrame.
        input_video_file: Path to the original video file.
        model_dir: Path to a trained LP model directory, or a list of paths
            for ensemble inference.
        output_base_dir: Base directory for all outputs.
        compute_metrics: If True, compute temporal norm and PCA reprojection
            error per animal. Useful for detecting tracking errors but adds
            overhead (rebuilds data module per animal). Default: False.

    Returns:
        Dict mapping animal_id → dict with paths to:
            - "cropped_video": cropped video file
            - "bbox_file": bbox CSV
            - "preds_file": raw predictions on cropped video (single model)
            - "remapped_preds_file": predictions in original coordinates (single model)
            - "ensemble_mean_file": ensemble-averaged predictions (ensemble only)
            - "ensemble_variance_file": per-keypoint variance (ensemble only)

    """
    # Lazy import to avoid circular dependency and heavy loading
    import gc

    import torch

    from lightning_pose.api.model import Model
    from lightning_pose.utils.predictions import compute_ensemble_variance

    # Determinism controls — ensure reproducible inference across runs
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Normalise model_dir to a list for uniform handling
    model_dirs = [model_dir] if isinstance(model_dir, Path) else list(model_dir)
    if len(model_dirs) == 0:
        raise ValueError("model_dir must contain at least one path")
    is_ensemble = len(model_dirs) > 1

    if len(model_dirs) > 20:
        raise ValueError(
            f"Ensemble with {len(model_dirs)} models is likely a mistake (max 20). "
            f"Check that model directories are listed correctly."
        )

    # Validate model dirs exist (fail fast before processing any animals)
    for d in model_dirs:
        if not d.is_dir():
            raise ValueError(f"Model directory does not exist: {d}")

    results: dict[str, dict[str, Path]] = {}

    # Pipeline completion sentinel
    sentinel = output_base_dir / "_RUNNING"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    sentinel.touch()

    # Write provenance metadata
    _write_provenance(output_base_dir, {
        "model_dir": [str(d) for d in model_dirs] if is_ensemble else str(model_dirs[0]),
        "video_file": str(input_video_file),
        "num_animals": len(bbox_dfs),
        "animal_ids": list(bbox_dfs.keys()),
        "ensemble_size": len(model_dirs),
    })

    try:
        for animal_id, bbox_df in bbox_dfs.items():
            safe_id = _sanitize_animal_id(animal_id)
            logger.info(f"Processing animal: {safe_id}")

            # Skip animals with all-NaN bboxes
            valid_frames = bbox_df.dropna().shape[0]
            if valid_frames == 0:
                logger.error(
                    f"Animal {safe_id}: all {len(bbox_df)} frames have invalid bboxes. "
                    f"Skipping this animal."
                )
                results[safe_id] = {"error": "no valid bboxes"}
                continue

            if valid_frames < len(bbox_df) * 0.5:
                logger.warning(
                    f"Animal {safe_id}: only {valid_frames}/{len(bbox_df)} frames "
                    f"have valid bboxes ({valid_frames / len(bbox_df):.0%})."
                )

            try:
                # Step 1: Crop video (once — bbox is the same for every model)
                crop_result = crop_video_for_animal(
                    animal_id=safe_id,
                    bbox_df=bbox_df,
                    input_video_file=input_video_file,
                    output_base_dir=output_base_dir,
                )

                cropped_video = crop_result["cropped_video"]
                animal_dir = output_base_dir / safe_id

                # Step 2: Run LP prediction with each model (loaded sequentially
                # to avoid GPU OOM with large ensembles)
                remapped_paths: list[Path] = []
                for model_idx, model_dir in enumerate(model_dirs):
                    model = None
                    try:
                        model = Model.from_dir(model_dir)
                        if is_ensemble:
                            pred_subdir = animal_dir / f"model_{model_idx}"
                            pred_subdir.mkdir(parents=True, exist_ok=True)
                        else:
                            pred_subdir = animal_dir

                        model.predict_on_video_file(
                            video_file=cropped_video,
                            output_dir=pred_subdir,
                            compute_metrics=compute_metrics,
                            generate_labeled_video=False,
                        )
                        preds_file = pred_subdir / f"{cropped_video.stem}.csv"

                        # Step 3: Remap to original coordinates
                        remapped_preds_file = remap_predictions(
                            preds_file=preds_file,
                            bbox_file=crop_result["bbox_file"],
                        )
                        remapped_paths.append(remapped_preds_file)
                    except (MemoryError, OSError):
                        raise
                    except Exception as e:
                        if is_ensemble:
                            logger.error(
                                f"Model {model_idx} failed for {safe_id}: {e}"
                            )
                            continue
                        raise
                    finally:
                        # Free GPU memory before loading next model
                        del model
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                # For single-model, keep the flat structure
                if not is_ensemble:
                    preds_file = animal_dir / f"{cropped_video.stem}.csv"
                    results[safe_id] = {
                        **crop_result,
                        "preds_file": preds_file,
                        "remapped_preds_file": remapped_paths[0],
                    }
                else:
                    # Need at least 2 successful models for ensemble variance
                    valid_paths = [p for p in remapped_paths if p.exists()]
                    if len(valid_paths) == 2:
                        logger.info(
                            f"Animal {safe_id}: ensemble with 2 models provides "
                            f"variance estimates but limited statistical power for QC. "
                            f"Consider 3+ models for robust QC."
                        )
                    if len(valid_paths) < 2:
                        logger.warning(
                            f"Animal {safe_id}: only {len(valid_paths)} model(s) "
                            f"succeeded, need >= 2 for ensemble variance"
                        )
                        results[safe_id] = {
                            **crop_result,
                            "error": f"only {len(valid_paths)} model(s) succeeded",
                        }
                        continue

                    # Step 4: Compute ensemble variance across models
                    mean_preds, variance_df = compute_ensemble_variance(valid_paths)

                    # Atomic write to prevent corruption from crash mid-write
                    ensemble_mean_file = animal_dir / "ensemble_mean.csv"
                    ensemble_variance_file = animal_dir / "ensemble_variance.csv"
                    tmp_mean = animal_dir / ".ensemble_mean.csv.tmp"
                    tmp_var = animal_dir / ".ensemble_variance.csv.tmp"
                    mean_preds.to_csv(tmp_mean)
                    variance_df.to_csv(tmp_var)
                    os.replace(tmp_mean, ensemble_mean_file)
                    os.replace(tmp_var, ensemble_variance_file)

                    results[safe_id] = {
                        **crop_result,
                        "ensemble_mean_file": ensemble_mean_file,
                        "ensemble_variance_file": ensemble_variance_file,
                    }

                # Mark per-animal completion
                (animal_dir / "_COMPLETE").touch()

            except (MemoryError, OSError):
                raise
            except Exception as e:
                logger.error(f"Failed to process animal {safe_id}: {e}")
                results[safe_id] = {"error": str(e)}

        any_failures = any("error" in v for v in results.values())
        target = "_PARTIAL" if any_failures else "_COMPLETE"
        sentinel.rename(output_base_dir / target)

        # Update provenance with per-animal completion status
        _write_provenance(output_base_dir, {
            "model_dir": [str(d) for d in model_dirs] if is_ensemble else str(model_dirs[0]),
            "video_file": str(input_video_file),
            "num_animals": len(bbox_dfs),
            "animal_ids": list(bbox_dfs.keys()),
            "ensemble_size": len(model_dirs),
            "animal_status": {
                aid: "failed" if "error" in paths else "success"
                for aid, paths in results.items()
            },
            "status": target.strip("_").lower(),
        })
    except Exception:
        sentinel.rename(output_base_dir / "_FAILED")
        raise

    return results


@typechecked
def merge_multi_animal_predictions(
    prediction_files: dict[str, Path],
    output_file: Path,
) -> pd.DataFrame:
    """Merge per-animal predictions into a single multi-animal CSV.

    Creates a CSV with a multi-level header that includes animal identity:
        (animal_id, bodypart, x/y)

    Note: The merged format uses an LP-specific 3-level column layout
    ``(animal, bodypart, coord)`` and is **not** DLC-compatible. Individual
    per-animal prediction CSVs (before merging) use the standard DLC format
    ``(scorer, bodypart, coord)`` and can be consumed by DLC tools directly.

    Args:
        prediction_files: Dict mapping animal_id → path to remapped predictions CSV.
        output_file: Where to save the merged CSV.

    Returns:
        The merged DataFrame.

    """
    dfs = {}
    for animal_id, pred_file in prediction_files.items():
        df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
        df = io.fix_empty_first_row(df)
        # Add animal_id as a top-level column index
        new_cols = pd.MultiIndex.from_tuples(
            [(animal_id, bp, coord) for _, bp, coord in df.columns],
            names=["animal", "bodyparts", "coords"],
        )
        df.columns = new_cols
        dfs[animal_id] = df

    merged = pd.concat(dfs.values(), axis=1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file)
    logger.info(f"Merged predictions for {len(dfs)} animals → {output_file}")

    return merged
