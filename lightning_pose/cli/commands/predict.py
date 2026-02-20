"""Predict command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lightning_pose.api.model import Model

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def register_parser(subparsers):
    """Register the predict command parser."""
    predict_parser = subparsers.add_parser(
        "predict",
        description=textwrap.dedent(
            """\
        Predicts keypoints on videos or images.

          Single model — predictions are saved to::

            <model_dir>/
            └── video_preds/
                ├── <video_filename>.csv              (predictions)
                ├── <video_filename>_<metric>.csv     (losses)
                └── labeled_videos/
                    └── <video_filename>_labeled.mp4

          Ensemble mode (multiple model_dirs) — predictions are saved to::

            <output_dir>/
            └── <video_stem>/
                ├── ensemble_mean.csv
                ├── ensemble_variance.csv
                ├── model_0/<video_stem>.csv
                ├── model_1/<video_stem>.csv
                └── ...

          Image predictions are saved to::

            <model_dir>/
            └── image_preds/
                └── <image_dirname | csv_filename | timestamp>/
                    ├── predictions.csv
                    ├── predictions_<metric>.csv      (losses)
                    └── <image_filename>_labeled.png
        """
        ),
        usage="litpose predict <model_dir>...  <input_path:video|image|dir|csv>...  [OPTIONS]",
    )
    predict_parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="model_dir/input_path",
        help="one or more model directories followed by input paths "
        "(video files, image files, CSV files, or directories). "
        "Multiple model directories enable ensemble mode.",
    )
    predict_parser.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="overrides attributes of the config file. Uses hydra syntax:\n"
        "https://hydra.cc/docs/advanced/override_grammar/basic/",
    )

    predict_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite videos that already have prediction files",
    )

    predict_parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="output directory for ensemble predictions (required in ensemble mode)",
    )

    post_prediction_args = predict_parser.add_argument_group("post-prediction")
    post_prediction_args.add_argument(
        "--skip_viz",
        action="store_true",
        help="skip generating prediction-annotated images/videos",
    )
    post_prediction_args.add_argument(
        "--skip_qc",
        action="store_true",
        help="skip QC flagging after ensemble prediction",
    )
    # For app use only.
    post_prediction_args.add_argument(
        "--progress_file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return predict_parser


def get_parser():
    """Return an ArgumentParser for the `litpose predict` subcommand (for docs)."""
    import argparse

    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


def handle(args):
    """Handle the predict command."""
    # All positional args come in as a single list. Partition into model dirs
    # (leading directories that contain a model config) and input paths (the rest).
    model_dirs: list[Path] = []
    input_paths: list[Path] = []
    found_non_model = False
    for p in args.paths:
        if not found_non_model and p.is_dir() and _looks_like_model_dir(p):
            model_dirs.append(p)
        else:
            found_non_model = True
            input_paths.append(p)

    if not model_dirs:
        raise SystemExit("Error: at least one model_dir is required")
    if not input_paths:
        raise SystemExit("Error: at least one input_path is required")

    if len(model_dirs) > 1:
        _handle_ensemble(args, model_dirs, input_paths)
    else:
        _handle_single(args, model_dirs[0], input_paths)


def _looks_like_model_dir(p: Path) -> bool:
    """Check if a directory looks like a trained LP model directory.

    Requires a config file AND at least one checkpoint or training artifact
    to avoid misclassifying data directories that contain config.yaml.

    Known edge cases that may cause false positives:
    - A data directory containing a stale ``tb_logs/`` folder and a
      ``config.yaml`` from a previous run will be treated as a model dir.
    - Directories with ``.ckpt`` files from non-LP frameworks will match
      if they also contain ``config.yaml``.

    When in doubt, list model directories before input paths and verify
    the partitioning in the console output.
    """
    has_config = (p / "config.yaml").exists() or (p / "training_config.yaml").exists()
    if not has_config:
        return False
    # Check for checkpoint files or training artifacts
    if next(p.glob("*.ckpt"), None) is not None:
        return True
    checkpoints_dir = p / "checkpoints"
    if checkpoints_dir.is_dir() and next(checkpoints_dir.glob("*.ckpt"), None) is not None:
        return True
    return False


def _handle_single(args, model_dir: Path, input_paths: list[Path]):
    """Single-model prediction — original behavior."""
    from lightning_pose.api.model import Model

    model = Model.from_dir2(model_dir, hydra_overrides=args.overrides)

    if model.config.is_multi_view():
        _predict_multi_type_multi_view(
            model, input_paths, args.skip_viz, not args.overwrite,
            progress_file=args.progress_file,
        )
    else:
        for p in input_paths:
            _predict_multi_type(
                model, p, args.skip_viz, not args.overwrite,
                progress_file=args.progress_file,
            )


def _handle_ensemble(args, model_dirs: list[Path], input_paths: list[Path]):
    """Ensemble prediction — multiple models, compute variance."""
    if args.output_dir is None:
        raise SystemExit(
            "Error: --output_dir is required for ensemble mode "
            "(multiple model directories)"
        )

    output_dir: Path = args.output_dir

    if len(model_dirs) > 20:
        raise SystemExit(
            f"Error: ensemble with {len(model_dirs)} models is likely a mistake "
            f"(max 20). Check that model directories are listed before input paths."
        )

    # Validate model dirs (fail fast without heavy model loading)
    if len(model_dirs) == 2:
        print(
            "NOTE: ensemble with 2 models provides variance estimates but "
            "limited statistical power for QC. Consider 3+ models for robust QC."
        )
    print(f"Ensemble mode: {len(model_dirs)} models")
    for i, d in enumerate(model_dirs):
        print(f"  [{i}] {d}")
        if not d.is_dir():
            raise SystemExit(f"Error: model directory does not exist: {d}")

    # Collect all video files first to check for stem collisions
    all_video_files: list[Path] = []
    for p in input_paths:
        if p.is_dir():
            video_files = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix in _VIDEO_EXTENSIONS
            )
            if not video_files:
                print(f"Skipping {p}: no video files found")
                continue
            print(f"Processing directory {p} ({len(video_files)} videos)")
            all_video_files.extend(video_files)
        elif p.suffix in _VIDEO_EXTENSIONS:
            all_video_files.append(p)
        else:
            print(
                f"Skipping {p}: ensemble mode only supports video files "
                f"({', '.join(sorted(_VIDEO_EXTENSIONS))})"
            )

    # Detect duplicate stems — would silently overwrite output
    stem_to_files: dict[str, list[Path]] = {}
    for vf in all_video_files:
        stem_to_files.setdefault(vf.stem, []).append(vf)
    dupes = {stem: paths for stem, paths in stem_to_files.items() if len(paths) > 1}
    if dupes:
        lines = [f"  {stem}: {', '.join(str(p) for p in paths)}" for stem, paths in dupes.items()]
        raise SystemExit(
            "Error: multiple videos share the same stem, which would overwrite "
            "ensemble output:\n" + "\n".join(lines)
        )

    # Process each video
    for vf in all_video_files:
        _predict_ensemble_video(model_dirs, vf, output_dir, args)


def _predict_ensemble_video(
    model_dirs: list[Path],
    video_file: Path,
    output_dir: Path,
    args,
):
    """Run all models on a single video and compute ensemble variance.

    Models are loaded and unloaded sequentially to avoid GPU OOM with
    large ensembles. Per-model errors are caught so that a single model
    failure does not discard predictions from successful models.
    """
    import gc
    import json
    import os
    import tempfile
    from datetime import datetime, timezone

    import torch

    from lightning_pose.api.model import Model
    from lightning_pose.utils.predictions import compute_ensemble_variance

    video_stem = video_file.stem
    video_out_dir = output_dir / video_stem

    # Check for existing ensemble output
    existing_mean = video_out_dir / "ensemble_mean.csv"
    if not args.overwrite and existing_mean.exists():
        print(f"Skipping {video_file.name} (ensemble_mean.csv already exists)")
        return

    print(f"\nEnsemble prediction: {video_file.name}")
    pred_csv_paths: list[Path] = []
    # Track per-model status for provenance
    model_status: list[dict] = []

    for model_idx, model_dir in enumerate(model_dirs):
        model_subdir = video_out_dir / f"model_{model_idx}"
        model_subdir.mkdir(parents=True, exist_ok=True)

        print(f"  Model {model_idx}/{len(model_dirs) - 1} ({model_dir.name})...")
        model = None
        try:
            model = Model.from_dir2(model_dir, hydra_overrides=args.overrides)
            if model.config.is_multi_view():
                raise SystemExit(
                    f"Error: ensemble mode does not support multi-view models "
                    f"(model {model_idx}: {model_dir})"
                )
            model.predict_on_video_file(
                video_file=video_file,
                output_dir=model_subdir,
                compute_metrics=False,
                generate_labeled_video=False,
            )
            pred_csv = model_subdir / f"{video_stem}.csv"
            pred_csv_paths.append(pred_csv)
            model_status.append({
                "index": model_idx,
                "model_dir": str(model_dir.resolve()),
                "status": "success",
            })
        except SystemExit:
            raise
        except (MemoryError, OSError):
            raise
        except Exception as e:
            print(f"    FAILED: {e}")
            model_status.append({
                "index": model_idx,
                "model_dir": str(model_dir.resolve()),
                "status": "failed",
                "error": str(e),
            })
            continue
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Need at least 2 successful predictions for ensemble variance
    valid_paths = [p for p in pred_csv_paths if p.exists()]
    if len(valid_paths) < 2:
        print(
            f"  WARNING: only {len(valid_paths)} model(s) produced predictions, "
            f"need >= 2 for ensemble variance"
        )
        return

    # Compute ensemble mean and variance
    print("  Computing ensemble variance...")
    mean_preds, variance_df = compute_ensemble_variance(valid_paths)

    # Atomic write: temp file + os.replace() to prevent silent corruption
    # from crash mid-write (existence-based skip would treat partial CSV as complete)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_mean_file = video_out_dir / "ensemble_mean.csv"
    ensemble_variance_file = video_out_dir / "ensemble_variance.csv"

    tmp_mean = video_out_dir / f".ensemble_mean.csv.tmp"
    tmp_var = video_out_dir / f".ensemble_variance.csv.tmp"
    mean_preds.to_csv(tmp_mean)
    variance_df.to_csv(tmp_var)
    os.replace(tmp_mean, ensemble_mean_file)
    os.replace(tmp_var, ensemble_variance_file)

    print(f"  Saved: {ensemble_mean_file}")
    print(f"  Saved: {ensemble_variance_file}")

    # Write provenance metadata
    import numpy as np
    import sys as _sys
    try:
        import lightning_pose
        lp_version = getattr(lightning_pose, "__version__", "unknown")
    except Exception:
        lp_version = "unknown"

    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_file": str(video_file.resolve()),
        "n_models_requested": len(model_dirs),
        "n_models_succeeded": sum(1 for m in model_status if m["status"] == "success"),
        "models": model_status,
        "versions": {
            "python": _sys.version,
            "numpy": np.__version__,
            "lightning_pose": lp_version,
        },
    }
    provenance_file = video_out_dir / "provenance.json"
    with open(provenance_file, "w") as f:
        json.dump(provenance, f, indent=2)

    # QC flagging on ensemble variance
    if not args.skip_qc:
        try:
            from lightning_pose.utils.qc import (
                flag_outlier_frames,
                format_qc_summary,
                save_qc_report,
            )

            qc_result = flag_outlier_frames(ensemble_variance_df=variance_df)
            save_qc_report(qc_result, video_out_dir)
            print(f"  {format_qc_summary(qc_result)}")
        except Exception as e:
            logger.debug("QC flagging failed", exc_info=True)
            print(
                f"  WARNING: QC flagging failed ({type(e).__name__}: {e}). "
                f"Predictions were saved successfully. "
                f"Use --skip_qc to suppress QC, or report this issue."
            )


def _predict_multi_type(
    model: Model,
    path: Path,
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
):
    if path.is_dir():
        image_files = [p for p in path.iterdir() if p.is_file() and p.suffix in [".png", ".jpg"]]
        video_files = [p for p in path.iterdir() if p.is_file() and p.suffix in _VIDEO_EXTENSIONS]

        if len(image_files) > 0:
            raise NotImplementedError("Predicting on image dir.")

        print(f"Processing directory {path}")
        for p in video_files:
            _predict_multi_type(model, p, skip_viz, skip_existing, progress_file=progress_file)

    elif path.suffix in _VIDEO_EXTENSIONS:
        # Check if prediction file already exists
        prediction_csv_file = model.video_preds_dir() / f"{path.stem}.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_video_file(
            video_file=path,
            generate_labeled_video=(not skip_viz),
            progress_file=progress_file,
        )
    elif path.suffix == ".csv":
        # Check if prediction file already exists
        prediction_csv_file = model.image_preds_dir() / path.name / "predictions.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_label_csv(
            csv_file=path,
        )
    elif path.suffix in [".png", ".jpg"]:
        raise NotImplementedError("Not yet implemented: predicting on image files.")
    else:
        pass


def _predict_multi_type_multi_view(
    model: Model,
    paths: list[Path],
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
):
    # delay this import because it's slow
    from lightning_pose.utils.io import (
        extract_session_name_from_video,
        split_video_files_by_view,
    )

    # if we pass in all videos, collect them into session batches and process
    if all(path.suffix in _VIDEO_EXTENSIONS for path in paths):
        video_files_split = split_video_files_by_view(
            paths, model.config.cfg.data.view_names
        )
        print(f"Grouped {len(paths)} videos into {len(video_files_split)} sessions:")
        pprint(
            [
                extract_session_name_from_video(
                    video_file_per_view[0].name, model.config.cfg.data.view_names
                )
                for video_file_per_view in video_files_split
            ],
        )
        for video_file_per_view in video_files_split:
            if skip_existing and all(
                (model.video_preds_dir() / f"{Path(video).stem}.csv").exists()
                for video in video_file_per_view
            ):
                session_name = extract_session_name_from_video(
                    Path(video_file_per_view[0]).name, model.config.cfg.data.view_names
                )
                print(f"Skipping {session_name} (prediction file already exists)")
                continue

            model.predict_on_video_file_multiview(
                video_file_per_view,
                generate_labeled_video=not skip_viz,
                progress_file=progress_file,
            )
    # if we have a list of directories, we process the videos in each separately
    elif all(path.is_dir() for path in paths):
        for path in paths:
            video_files = [
                p for p in path.iterdir() if p.is_file() and p.suffix in _VIDEO_EXTENSIONS
            ]
            if len(video_files) > 0:
                print(f"Processing directory {path}")

                _predict_multi_type_multi_view(
                    model, video_files, skip_viz, skip_existing, progress_file=progress_file
                )
            else:
                print(f"Skipping {path}: no videos found.")
    else:
        raise NotImplementedError(
            "For multi view model predictions, either pass in multiple video views to be predicted, or a directory containing videos"
        )
