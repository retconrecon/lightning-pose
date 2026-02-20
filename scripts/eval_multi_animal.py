"""Evaluation utilities for multi-animal SAM→LP predictions.

Computes per-keypoint, per-animal Euclidean distance between
predicted and ground truth keypoints.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ground_truth(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a ground truth CSV with 3-row DLC header.

    Expects columns like (scorer, bodypart_name, x/y/likelihood).
    Bodypart names should be prefixed with animal identity
    (e.g., black_mouse_nose, white_mouse_nose).

    Returns:
        (dataframe, list of unique bodypart names, list of coord types)

    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    coords = df.columns.get_level_values(2).unique().tolist()
    return df, bodyparts, coords


def split_bodyparts_by_animal(
    bodyparts: list[str],
    animal_prefixes: list[str] | None = None,
) -> dict[str, list[str]]:
    """Split bodypart names into per-animal groups.

    If animal_prefixes is None, infers prefixes by finding the longest
    common prefix shared by groups of bodyparts.

    Args:
        bodyparts: e.g., ["black_mouse_nose", "black_mouse_ear", "white_mouse_nose", ...]
        animal_prefixes: e.g., ["black_mouse", "white_mouse"]

    Returns:
        Dict mapping animal_prefix → list of bodypart names.

    """
    if animal_prefixes is None:
        # Heuristic: find unique prefixes by splitting on last underscore groups
        # Try progressively shorter prefixes until we get groups
        prefixes = set()
        for bp in bodyparts:
            parts = bp.split("_")
            # Try 2-word prefix first, then 1-word
            for n_prefix_words in range(len(parts) - 1, 0, -1):
                prefix = "_".join(parts[:n_prefix_words])
                # Check if this prefix is shared by at least 2 bodyparts
                matches = [b for b in bodyparts if b.startswith(prefix + "_")]
                if len(matches) >= 2:
                    prefixes.add(prefix)
                    break

        # Keep only prefixes that aren't substrings of other prefixes
        animal_prefixes = sorted(
            [p for p in prefixes if not any(
                p != q and p.startswith(q) for q in prefixes
            )],
        )

    # Fallback: if no prefixes found, treat all bodyparts as a single animal
    if not animal_prefixes:
        logger.warning(
            f"Could not infer animal prefixes from bodypart names: {bodyparts}. "
            f"Treating all bodyparts as belonging to a single animal."
        )
        return {"animal": bodyparts}

    result = {}
    for prefix in animal_prefixes:
        result[prefix] = [bp for bp in bodyparts if bp.startswith(prefix + "_")]
    return result


def compute_euclidean_distances(
    preds_csv: str | Path,
    gt_csv: str | Path,
    animal_prefixes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute per-frame, per-keypoint Euclidean distance between predictions and ground truth.

    Args:
        preds_csv: Path to predictions CSV (DLC 3-row header format with x, y columns).
        gt_csv: Path to ground truth CSV (DLC 3-row header with x, y, likelihood columns).
        animal_prefixes: Optional list of animal name prefixes.
            If None, auto-detected from bodypart names.

    Returns:
        Dict mapping animal_prefix → DataFrame of shape (num_frames, num_keypoints)
        with Euclidean distances per frame per keypoint.

    """
    # Load both CSVs
    pred_df = pd.read_csv(preds_csv, header=[0, 1, 2], index_col=0)
    gt_df = pd.read_csv(gt_csv, header=[0, 1, 2], index_col=0)

    # Align frames
    n_frames = min(len(pred_df), len(gt_df))
    if len(pred_df) != len(gt_df):
        logger.warning(
            f"Frame count mismatch: preds={len(pred_df)}, gt={len(gt_df)}. "
            f"Using first {n_frames} frames."
        )
    pred_df = pred_df.iloc[:n_frames]
    gt_df = gt_df.iloc[:n_frames]

    # Get bodyparts from ground truth
    gt_bodyparts = gt_df.columns.get_level_values(1).unique().tolist()
    animals = split_bodyparts_by_animal(gt_bodyparts, animal_prefixes)

    # Check if predictions are in merged format (animal_id at level 0)
    pred_top_levels = pred_df.columns.get_level_values(0).unique().tolist()
    is_merged = any(lvl in pred_top_levels for lvl in animals.keys())

    results = {}
    for animal_id, bodypart_names in animals.items():
        distances = {}

        # If merged format, scope predictions to this animal first
        if is_merged and animal_id in pred_top_levels:
            animal_pred_df = pred_df.xs(animal_id, level=0, axis=1)
        else:
            animal_pred_df = pred_df

        pred_bodyparts = animal_pred_df.columns.get_level_values(
            0 if animal_pred_df.columns.nlevels == 2 else 1
        ).unique().tolist()

        for bp in bodypart_names:
            # Extract x, y from ground truth
            gt_x = gt_df.xs(bp, level=1, axis=1).xs("x", level=1, axis=1).values.flatten()
            gt_y = gt_df.xs(bp, level=1, axis=1).xs("y", level=1, axis=1).values.flatten()

            # Try to find matching bodypart in predictions
            short_name = bp[len(animal_id) + 1:] if animal_id else bp
            target_bp = None
            if bp in pred_bodyparts:
                target_bp = bp
            elif short_name in pred_bodyparts:
                target_bp = short_name
            else:
                logger.warning(
                    f"Bodypart '{bp}' (or '{short_name}') not found in predictions"
                )
                distances[bp] = np.full(n_frames, np.nan)
                continue

            # Extract pred x, y — handle different column depths
            if animal_pred_df.columns.nlevels == 2:
                # 2-level: (bodypart, coord) — after xs on animal level
                bp_df = animal_pred_df.xs(target_bp, level=0, axis=1)
                pred_x = bp_df["x"].values.flatten()
                pred_y = bp_df["y"].values.flatten()
            else:
                # 3-level: (scorer, bodypart, coord) — standard DLC format
                pred_x = animal_pred_df.xs(target_bp, level=1, axis=1).xs(
                    "x", level=1, axis=1
                ).values.flatten()
                pred_y = animal_pred_df.xs(target_bp, level=1, axis=1).xs(
                    "y", level=1, axis=1
                ).values.flatten()

            # Shape guard: ensure pred and gt arrays match
            if len(pred_x) != n_frames or len(gt_x) != n_frames:
                logger.warning(
                    f"Shape mismatch for '{bp}': pred_x={len(pred_x)}, "
                    f"gt_x={len(gt_x)}, n_frames={n_frames}. Skipping."
                )
                distances[bp] = np.full(n_frames, np.nan)
                continue

            # Euclidean distance
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            distances[bp] = dist

        results[animal_id] = pd.DataFrame(distances, index=range(n_frames))

    return results


def summarize_distances(
    distances: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize per-keypoint Euclidean distances.

    Args:
        distances: Output of compute_euclidean_distances().

    Returns:
        DataFrame with columns: animal, keypoint, mean, median, std, p50, p90, p95

    """
    rows = []
    for animal_id, dist_df in distances.items():
        for keypoint in dist_df.columns:
            vals = dist_df[keypoint].dropna()
            if len(vals) == 0:
                rows.append({
                    "animal": animal_id,
                    "keypoint": keypoint,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                    "p95": np.nan,
                    "n_frames": 0,
                })
                continue
            rows.append({
                "animal": animal_id,
                "keypoint": keypoint,
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "p50": np.percentile(vals, 50),
                "p90": np.percentile(vals, 90),
                "p95": np.percentile(vals, 95),
                "n_frames": len(vals),
            })

    # Add overall per-animal summary
    for animal_id, dist_df in distances.items():
        all_vals = dist_df.values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            rows.append({
                "animal": animal_id,
                "keypoint": "_ALL_",
                "mean": np.mean(all_vals),
                "median": np.median(all_vals),
                "std": np.std(all_vals),
                "p50": np.percentile(all_vals, 50),
                "p90": np.percentile(all_vals, 90),
                "p95": np.percentile(all_vals, 95),
                "n_frames": len(all_vals),
            })

    return pd.DataFrame(rows)


def evaluate(
    preds_csv: str | Path,
    gt_csv: str | Path,
    animal_prefixes: list[str] | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Full evaluation: compute distances and print summary.

    Args:
        preds_csv: Path to predictions CSV.
        gt_csv: Path to ground truth CSV.
        animal_prefixes: Optional animal name prefixes.
        output_csv: Optional path to save detailed distances.

    Returns:
        Summary DataFrame.

    """
    distances = compute_euclidean_distances(preds_csv, gt_csv, animal_prefixes)
    summary = summarize_distances(distances)

    # Print summary
    print("\n" + "=" * 80)
    print("Multi-Animal Pose Estimation Evaluation")
    print("=" * 80)

    for animal_id in distances:
        animal_rows = summary[
            (summary["animal"] == animal_id) & (summary["keypoint"] != "_ALL_")
        ]
        overall = summary[
            (summary["animal"] == animal_id) & (summary["keypoint"] == "_ALL_")
        ]

        print(f"\n--- {animal_id} ---")
        for _, row in animal_rows.iterrows():
            short_name = row["keypoint"]
            print(
                f"  {short_name:40s}  "
                f"mean={row['mean']:7.2f}  "
                f"median={row['median']:7.2f}  "
                f"p90={row['p90']:7.2f}  "
                f"p95={row['p95']:7.2f}"
            )
        if len(overall) > 0:
            o = overall.iloc[0]
            print(
                f"  {'OVERALL':40s}  "
                f"mean={o['mean']:7.2f}  "
                f"median={o['median']:7.2f}  "
                f"p90={o['p90']:7.2f}  "
                f"p95={o['p95']:7.2f}"
            )

    print()

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
        print(f"Summary saved to {output_csv}")

    return summary
