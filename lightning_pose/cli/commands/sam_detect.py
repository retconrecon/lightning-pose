"""SAM-based detection command for the lightning-pose CLI.

Provides two subcommands:
- `litpose sam-detect prepare-dataset`: Convert SAM masks → per-animal cropped datasets
- `litpose sam-detect infer`: Run multi-animal inference with SAM-derived bboxes
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def register_parser(subparsers):
    """Register the sam-detect command parser."""
    import sys

    is_building_docs = "sphinx" in sys.modules
    _doc_link = (
        ":doc:`SAM Detection </source/user_guide_advanced/sam_detection>`"
        if is_building_docs
        else "https://lightning-pose.readthedocs.io/en/latest/source/user_guide_advanced/sam_detection.html"
    )

    description_text = dedent(
        f"""\
        SAM3→SAM2 integration for multi-animal pose estimation.

        This command converts SAM segmentation masks into per-animal cropped datasets
        or runs multi-animal inference using SAM-derived bounding boxes.

        **Prepare dataset** — converts SAM masks to per-animal cropped datasets::

            litpose sam-detect prepare-dataset \\
                --masks_dir <path/to/masks/> \\
                --data_dir <path/to/data/> \\
                --csv_file <CollectedData.csv> \\
                --output_dir <path/to/output/>

        **Infer** — runs LP inference per animal on a new video::

            litpose sam-detect infer \\
                --bbox_dir <path/to/bbox_csvs/> \\
                --video <path/to/video.mp4> \\
                --model_dir <path/to/model/> \\
                --output_dir <path/to/output/>

        Mask files should be NumPy .npy files with shape (num_frames, H, W),
        one per animal, named like ``animal_0.npy``, ``animal_1.npy``, etc.

        Alternatively, provide pre-computed bbox CSV files (one per animal) with
        columns [x, y, h, w].
        """
    )

    sam_parser = subparsers.add_parser(
        "sam-detect",
        description=description_text,
        usage="litpose sam-detect <action> [OPTIONS]",
    )

    sam_subparsers = sam_parser.add_subparsers(
        dest="sam_action",
        help="SAM detection action to perform.",
    )

    # -- prepare-dataset subcommand --
    prep_parser = sam_subparsers.add_parser(
        "prepare-dataset",
        description="Convert SAM masks to per-animal cropped datasets for training.",
        usage="litpose sam-detect prepare-dataset [OPTIONS]",
    )
    prep_input = prep_parser.add_argument_group("input")
    prep_input.add_argument(
        "--masks_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal mask .npy files. "
        "Each file should contain an array of shape (num_frames, H, W).",
    )
    prep_input.add_argument(
        "--bbox_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal bbox CSV files (alternative to masks). "
        "Each CSV should have columns [x, y, h, w] indexed by image path.",
    )
    prep_input.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the LP data directory containing labeled-data/ and images.",
    )
    prep_input.add_argument(
        "--csv_file",
        type=Path,
        required=True,
        help="Path to the labels CSV (DLC format).",
    )

    prep_output = prep_parser.add_argument_group("output")
    prep_output.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base output directory. Per-animal subdirectories will be created.",
    )

    prep_options = prep_parser.add_argument_group("options")
    prep_options.add_argument(
        "--crop_ratio",
        type=float,
        default=2.0,
        help="Factor to expand bounding box around animal. Default: 2.0.",
    )
    prep_options.add_argument(
        "--apply_masks",
        action="store_true",
        help="Zero out background pixels using masks before cropping (removes distractors).",
    )

    # -- prepare-labeling subcommand --
    label_parser = sam_subparsers.add_parser(
        "prepare-labeling",
        description="Crop frames per animal for easier pose annotation. "
        "Run this before labeling — no trained model or labels CSV needed.",
        usage="litpose sam-detect prepare-labeling [OPTIONS]",
    )
    label_input = label_parser.add_argument_group("input")
    label_input.add_argument(
        "--masks_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal mask .npy files.",
    )
    label_input.add_argument(
        "--bbox_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal bbox CSV files (alternative to masks).",
    )
    label_input.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to directory containing the full-frame images.",
    )
    label_output = label_parser.add_argument_group("output")
    label_output.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base output directory. Per-animal subdirectories will be created.",
    )
    label_options = label_parser.add_argument_group("options")
    label_options.add_argument(
        "--crop_ratio",
        type=float,
        default=2.0,
        help="Factor to expand bounding box around mask. Default: 2.0.",
    )
    label_options.add_argument(
        "--expand",
        type=float,
        default=1.0,
        help="Additional expansion factor on top of crop_ratio. "
        "Use >1.0 if keypoints are getting cut off. Default: 1.0.",
    )

    # -- infer subcommand --
    infer_parser = sam_subparsers.add_parser(
        "infer",
        description="Run multi-animal inference using SAM-derived bboxes.",
        usage="litpose sam-detect infer [OPTIONS]",
    )
    infer_input = infer_parser.add_argument_group("input")
    infer_input.add_argument(
        "--masks_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal mask .npy files for the video.",
    )
    infer_input.add_argument(
        "--bbox_dir",
        type=Path,
        default=None,
        help="Directory containing per-animal bbox CSV files (alternative to masks).",
    )
    infer_input.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the video file.",
    )
    infer_input.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to a trained LP model directory.",
    )

    infer_output = infer_input.add_argument_group("output")
    infer_output.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base output directory for per-animal predictions. "
        "Use a unique directory per video to avoid collisions in batch jobs.",
    )

    infer_options = infer_parser.add_argument_group("options")
    infer_options.add_argument(
        "--crop_ratio",
        type=float,
        default=2.0,
        help="Factor to expand bounding box around animal. Default: 2.0.",
    )
    infer_options.add_argument(
        "--merge",
        action="store_true",
        help="Merge all per-animal predictions into a single CSV.",
    )
    infer_options.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Compute temporal norm and PCA error per animal. "
        "Useful for detecting tracking errors. Adds ~10-20s overhead per animal.",
    )

    return sam_parser


def get_parser():
    """Return an ArgumentParser for the `litpose sam-detect` subcommand (for docs)."""
    import argparse

    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


def handle(args):
    """Handle the sam-detect command."""
    if args.sam_action == "prepare-dataset":
        _handle_prepare_dataset(args)
    elif args.sam_action == "prepare-labeling":
        _handle_prepare_labeling(args)
    elif args.sam_action == "infer":
        _handle_infer(args)
    else:
        print("Please specify an action: 'prepare-dataset', 'prepare-labeling', or 'infer'")
        print("Usage: litpose sam-detect <action> --help")


def _load_bbox_dfs_from_masks(masks_dir: Path, crop_ratio: float) -> dict[str, "pd.DataFrame"]:
    """Load mask .npy files and convert to bbox DataFrames."""
    import numpy as np

    from lightning_pose.utils.sam import _sanitize_animal_id, sam_masks_to_lp_bbox

    mask_files = sorted(masks_dir.glob("*.npy"))
    if not mask_files:
        raise FileNotFoundError(f"No .npy mask files found in {masks_dir}")

    bbox_dfs = {}
    for mask_file in mask_files:
        animal_id = _sanitize_animal_id(mask_file.stem)
        masks = np.load(mask_file, mmap_mode="r", allow_pickle=False)
        # Use integer indices to match LP prediction output indices
        image_index = list(range(masks.shape[0]))
        bbox_dfs[animal_id] = sam_masks_to_lp_bbox(masks, image_index, crop_ratio=crop_ratio)

    print(f"Loaded masks for {len(bbox_dfs)} animals from {masks_dir}")
    if len(bbox_dfs) > 10:
        print(
            f"WARNING: Found {len(bbox_dfs)} animals — this is unusually high. "
            f"Check for false detections (reflections, cage hardware, etc.)."
        )
    return bbox_dfs


def _load_bbox_dfs_from_csvs(bbox_dir: Path) -> dict[str, "pd.DataFrame"]:
    """Load pre-computed bbox CSV files."""
    import pandas as pd

    from lightning_pose.utils.sam import _sanitize_animal_id, _validate_bbox_df

    bbox_files = sorted(bbox_dir.glob("*.csv"))
    if not bbox_files:
        raise FileNotFoundError(f"No .csv bbox files found in {bbox_dir}")

    bbox_dfs = {}
    for bbox_file in bbox_files:
        animal_id = _sanitize_animal_id(bbox_file.stem)
        df = pd.read_csv(bbox_file, index_col=0)
        _validate_bbox_df(df)
        bbox_dfs[animal_id] = df

    print(f"Loaded bboxes for {len(bbox_dfs)} animals from {bbox_dir}")
    return bbox_dfs


def _load_masks_per_animal(masks_dir: Path) -> dict[str, "np.ndarray"]:
    """Load mask arrays for background removal."""
    import numpy as np

    from lightning_pose.utils.sam import _sanitize_animal_id

    masks = {}
    for mask_file in sorted(masks_dir.glob("*.npy")):
        animal_id = _sanitize_animal_id(mask_file.stem)
        masks[animal_id] = np.load(mask_file, mmap_mode="r", allow_pickle=False)
    return masks


def _handle_prepare_labeling(args):
    """Crop frames per animal for pose annotation."""
    from lightning_pose.utils.sam import crop_frames_for_labeling

    # Resolve paths
    if args.masks_dir is not None:
        args.masks_dir = args.masks_dir.resolve()
    if args.bbox_dir is not None:
        args.bbox_dir = args.bbox_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    # Validate crop_ratio
    if not (0.5 <= args.crop_ratio <= 10.0):
        raise ValueError(
            f"crop_ratio must be between 0.5 and 10.0, got {args.crop_ratio}"
        )

    # Load bboxes
    if args.masks_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_masks(args.masks_dir, args.crop_ratio)
    elif args.bbox_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_csvs(args.bbox_dir)
    else:
        raise ValueError("Must provide either --masks_dir or --bbox_dir")

    # Crop frames per animal
    for animal_id, bbox_df in bbox_dfs.items():
        animal_output = args.output_dir / animal_id
        bbox_file = crop_frames_for_labeling(
            bbox_df=bbox_df,
            input_dir=args.data_dir,
            output_dir=animal_output,
            expand=args.expand,
        )
        print(f"  {animal_id}: cropped frames → {animal_output}")
        print(f"    bbox CSV: {bbox_file}")

    print(f"\nCropped frames for {len(bbox_dfs)} animals ready for labeling.")
    print("To adjust crop size, re-run with a different --expand value.")


def _handle_prepare_dataset(args):
    """Prepare per-animal cropped datasets from SAM masks or bboxes."""
    import pandas as pd

    from lightning_pose.utils.sam import prepare_all_animal_datasets

    # Resolve all paths to absolute for consistent downstream behavior
    if args.masks_dir is not None:
        args.masks_dir = args.masks_dir.resolve()
    if args.bbox_dir is not None:
        args.bbox_dir = args.bbox_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.csv_file = args.csv_file.resolve()
    args.output_dir = args.output_dir.resolve()

    # Validate crop_ratio
    if hasattr(args, "crop_ratio") and not (0.5 <= args.crop_ratio <= 10.0):
        raise ValueError(
            f"crop_ratio must be between 0.5 and 10.0, got {args.crop_ratio}"
        )

    # Load bboxes from masks or CSV files
    if args.masks_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_masks(args.masks_dir, args.crop_ratio)
    elif args.bbox_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_csvs(args.bbox_dir)
    else:
        raise ValueError("Must provide either --masks_dir or --bbox_dir")

    # Load image paths from the labels CSV to use as index
    csv_data = pd.read_csv(args.csv_file, header=[0, 1, 2], index_col=0)
    image_paths = list(csv_data.index)

    # Re-index bbox DataFrames to use image paths if they were from masks
    if args.masks_dir is not None:
        for animal_id in bbox_dfs:
            df = bbox_dfs[animal_id]
            if len(df) != len(image_paths):
                raise ValueError(
                    f"Mask for {animal_id} has {len(df)} frames but CSV has {len(image_paths)} rows"
                )
            df.index = image_paths
            bbox_dfs[animal_id] = df

    # Optionally load masks for background removal
    masks_per_animal = None
    if args.apply_masks and args.masks_dir is not None:
        masks_per_animal = _load_masks_per_animal(args.masks_dir)

    results = prepare_all_animal_datasets(
        bbox_dfs=bbox_dfs,
        input_data_dir=args.data_dir,
        input_csv_file=args.csv_file,
        output_base_dir=args.output_dir,
        masks_per_animal=masks_per_animal,
        image_paths=image_paths if masks_per_animal else None,
    )

    print("\nPer-animal datasets created:")
    for animal_id, paths in results.items():
        print(f"  {animal_id}:")
        for key, path in paths.items():
            print(f"    {key}: {path}")

    print(f"\nTo train a model on a specific animal, use:")
    first_animal = next(iter(results))
    r = results[first_animal]
    print(f"  litpose train <config.yaml> data.data_dir={r['data_dir']} data.csv_file={r['csv_file']}")


def _handle_infer(args):
    """Run multi-animal inference."""
    from lightning_pose.utils.sam import merge_multi_animal_predictions, run_multi_animal_inference

    # Resolve all paths to absolute for consistent downstream behavior
    if args.masks_dir is not None:
        args.masks_dir = args.masks_dir.resolve()
    if args.bbox_dir is not None:
        args.bbox_dir = args.bbox_dir.resolve()
    args.video = args.video.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    # Validate crop_ratio
    if hasattr(args, "crop_ratio") and not (0.5 <= args.crop_ratio <= 10.0):
        raise ValueError(
            f"crop_ratio must be between 0.5 and 10.0, got {args.crop_ratio}"
        )

    # Load bboxes from masks or CSV files
    if args.masks_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_masks(args.masks_dir, args.crop_ratio)
    elif args.bbox_dir is not None:
        bbox_dfs = _load_bbox_dfs_from_csvs(args.bbox_dir)
    else:
        raise ValueError("Must provide either --masks_dir or --bbox_dir")

    results = run_multi_animal_inference(
        bbox_dfs=bbox_dfs,
        input_video_file=args.video,
        model_dir=args.model_dir,
        output_base_dir=args.output_dir,
        compute_metrics=args.compute_metrics,
    )

    print("\nPer-animal inference results:")
    for animal_id, paths in results.items():
        print(f"  {animal_id}:")
        if "error" in paths:
            print(f"    FAILED: {paths['error']}")
        else:
            print(f"    remapped predictions: {paths['remapped_preds_file']}")

    if args.merge:
        # Skip failed animals when merging
        pred_files = {
            animal_id: paths["remapped_preds_file"]
            for animal_id, paths in results.items()
            if "remapped_preds_file" in paths
        }
        if not pred_files:
            print("\nERROR: No animals completed successfully. Cannot merge.")
        else:
            if len(pred_files) < len(results):
                failed = len(results) - len(pred_files)
                print(f"\nWARNING: {failed} animal(s) failed — merging {len(pred_files)} successful.")
            merged_file = args.output_dir / "merged_predictions.csv"
            merge_multi_animal_predictions(pred_files, merged_file)
            print(f"\nMerged predictions: {merged_file}")
