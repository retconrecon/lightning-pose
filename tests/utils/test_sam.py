"""Tests for SAM integration utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lightning_pose.utils.sam import (
    _sanitize_animal_id,
    _validate_bbox_df,
    apply_mask_to_images,
    merge_multi_animal_predictions,
    prepare_animal_dataset,
    sam_masks_to_lp_bbox,
    sam_masks_to_lp_bbox_batch,
)
# sam_eval lives in scripts/ as a standalone tool (no LP imports)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eval_multi_animal import (
    compute_euclidean_distances,
    split_bodyparts_by_animal,
    summarize_distances,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_masks():
    """Create simple binary masks for testing."""
    masks = np.zeros((5, 100, 100), dtype=np.uint8)
    # Animal occupies a 20x30 region starting at (10, 20)
    for i in range(5):
        masks[i, 20:50, 10:40] = 1  # y=20..50, x=10..40
    return masks


@pytest.fixture
def image_index():
    return [f"labeled-data/img{i:04d}.png" for i in range(5)]


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a minimal LP data directory with images and CSV."""
    data_dir = tmp_path / "data"
    labeled_dir = data_dir / "labeled-data"
    labeled_dir.mkdir(parents=True)

    # Create 5 simple 100x100 RGB images
    from PIL import Image

    image_paths = []
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_path = labeled_dir / f"img{i:04d}.png"
        img.save(img_path)
        image_paths.append(f"labeled-data/img{i:04d}.png")

    # Create a DLC-format CSV with 3-row header
    # 2 bodyparts, each with x,y
    header = pd.MultiIndex.from_tuples(
        [
            ("scorer", "bp1", "x"),
            ("scorer", "bp1", "y"),
            ("scorer", "bp2", "x"),
            ("scorer", "bp2", "y"),
        ]
    )
    data = np.random.rand(5, 4) * 80 + 10  # coordinates in [10, 90]
    csv_data = pd.DataFrame(data, index=image_paths, columns=header)
    csv_path = data_dir / "CollectedData.csv"
    csv_data.to_csv(csv_path)

    return data_dir, csv_path, image_paths


# ---------------------------------------------------------------------------
# Tests: _sanitize_animal_id
# ---------------------------------------------------------------------------


class TestSanitizeAnimalId:

    def test_clean_id_unchanged(self):
        assert _sanitize_animal_id("animal_0") == "animal_0"
        assert _sanitize_animal_id("mouse-1") == "mouse-1"
        assert _sanitize_animal_id("Fish2") == "Fish2"

    def test_path_traversal_sanitized(self):
        result = _sanitize_animal_id("../../etc/cron.d/backdoor")
        assert "/" not in result
        assert ".." not in result
        assert result == "______etc_cron_d_backdoor"

    def test_dots_sanitized(self):
        result = _sanitize_animal_id("animal.0")
        assert "." not in result
        assert result == "animal_0"

    def test_dots_become_underscores(self):
        """Dots are replaced with underscores, result is non-empty."""
        assert _sanitize_animal_id("...") == "___"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty after sanitization"):
            _sanitize_animal_id("")


# ---------------------------------------------------------------------------
# Tests: _validate_bbox_df
# ---------------------------------------------------------------------------


class TestValidateBboxDf:

    def test_valid_bbox_passes(self):
        df = pd.DataFrame(
            {"x": [10, 20], "y": [30, 40], "h": [50, 60], "w": [50, 60]}
        )
        _validate_bbox_df(df)  # should not raise

    def test_all_nan_passes(self):
        df = pd.DataFrame(
            {"x": [np.nan], "y": [np.nan], "h": [np.nan], "w": [np.nan]}
        )
        _validate_bbox_df(df)  # should not raise (caller handles all-NaN)

    def test_inf_raises(self):
        df = pd.DataFrame(
            {"x": [10], "y": [30], "h": [np.inf], "w": [50]}
        )
        with pytest.raises(ValueError, match="Inf"):
            _validate_bbox_df(df)

    def test_non_positive_dims_raises(self):
        df = pd.DataFrame(
            {"x": [10], "y": [30], "h": [0], "w": [50]}
        )
        with pytest.raises(ValueError, match="non-positive"):
            _validate_bbox_df(df)

    def test_mixed_nan_and_valid(self):
        df = pd.DataFrame(
            {"x": [10, np.nan], "y": [30, np.nan], "h": [50, np.nan], "w": [50, np.nan]}
        )
        _validate_bbox_df(df)  # should not raise


# ---------------------------------------------------------------------------
# Tests: sam_masks_to_lp_bbox
# ---------------------------------------------------------------------------


class TestSamMasksToLpBbox:

    def test_basic_conversion(self, sample_masks, image_index):
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=1.0)

        assert isinstance(bbox_df, pd.DataFrame)
        assert list(bbox_df.columns) == ["x", "y", "h", "w"]
        assert len(bbox_df) == 5
        assert list(bbox_df.index) == image_index

    def test_bbox_contains_mask(self, sample_masks, image_index):
        """Bbox should fully contain the mask region."""
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=1.0)

        for i in range(5):
            row = bbox_df.iloc[i]
            # Mask spans x=[10,39] (inclusive), y=[20,49] (inclusive)
            assert row.x <= 10, f"bbox x={row.x} should be <= 10"
            assert row.y <= 20, f"bbox y={row.y} should be <= 20"
            assert row.x + row.w >= 39, f"bbox right edge should be >= 39"
            assert row.y + row.h >= 49, f"bbox bottom edge should be >= 49"

    def test_crop_ratio_expands_bbox(self, sample_masks, image_index):
        bbox_small = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=1.0)
        bbox_large = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=2.0)

        assert (bbox_large["h"].values >= bbox_small["h"].values).all()
        assert (bbox_large["w"].values >= bbox_small["w"].values).all()

    def test_square_bboxes(self, sample_masks, image_index):
        """Bboxes should be square (h == w)."""
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=1.5)
        assert (bbox_df["h"] == bbox_df["w"]).all()

    def test_even_dimensions(self, sample_masks, image_index):
        """Bbox dimensions should be even (for video players)."""
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=1.3)
        # Only check non-NaN rows
        non_nan = bbox_df.dropna()
        assert (non_nan["h"] % 2 == 0).all()
        assert (non_nan["w"] % 2 == 0).all()

    def test_empty_mask_produces_nan(self, image_index):
        """Empty masks should produce NaN bboxes (not centered fallback)."""
        # Only 2 of 5 frames empty — under 50% threshold
        masks = np.zeros((5, 100, 100), dtype=np.uint8)
        masks[0, 20:50, 10:40] = 1
        masks[1, 20:50, 10:40] = 1
        masks[2, 20:50, 10:40] = 1
        # frames 3 and 4 are empty

        bbox_df = sam_masks_to_lp_bbox(masks, image_index, crop_ratio=1.0)
        assert len(bbox_df) == 5
        # First 3 frames should have valid bboxes
        assert not bbox_df.iloc[:3].isnull().any().any()
        # Last 2 frames should be NaN
        assert bbox_df.iloc[3:].isnull().all().all()

    def test_majority_empty_masks_raises(self, image_index):
        """More than 50% empty masks should raise ValueError."""
        masks = np.zeros((5, 100, 100), dtype=np.uint8)
        # Only 1 of 5 frames has content — 80% empty
        masks[0, 20:50, 10:40] = 1

        with pytest.raises(ValueError, match="50%"):
            sam_masks_to_lp_bbox(masks, image_index, crop_ratio=1.0)

    def test_wrong_ndim_raises(self, image_index):
        masks_2d = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 dims"):
            sam_masks_to_lp_bbox(masks_2d, image_index)

    def test_mismatched_lengths_raises(self, sample_masks):
        short_index = ["img0.png", "img1.png"]
        with pytest.raises(ValueError, match="image_index length"):
            sam_masks_to_lp_bbox(sample_masks, short_index)

    def test_nan_in_masks_raises(self, image_index):
        """Masks containing NaN should raise."""
        masks = np.zeros((5, 100, 100), dtype=np.float32)
        masks[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            sam_masks_to_lp_bbox(masks, image_index)

    def test_crop_ratio_too_low_raises(self, sample_masks, image_index):
        with pytest.raises(ValueError, match="crop_ratio"):
            sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=0.1)

    def test_crop_ratio_too_high_raises(self, sample_masks, image_index):
        with pytest.raises(ValueError, match="crop_ratio"):
            sam_masks_to_lp_bbox(sample_masks, image_index, crop_ratio=100.0)

    def test_integer_index_from_masks(self):
        """When using integer indices (like from CLI), index should be int."""
        masks = np.zeros((3, 50, 50), dtype=np.uint8)
        masks[:, 10:30, 10:30] = 1
        int_index = list(range(3))
        bbox_df = sam_masks_to_lp_bbox(masks, int_index, crop_ratio=1.5)
        # Index should be integers, not strings
        assert bbox_df.index.tolist() == [0, 1, 2]
        assert isinstance(bbox_df.index[0], (int, np.integer))


class TestSamMasksToLpBboxBatch:

    def test_multi_animal(self, sample_masks, image_index):
        masks_per_animal = {
            "mouse_0": sample_masks,
            "mouse_1": np.roll(sample_masks, 20, axis=2),
        }
        result = sam_masks_to_lp_bbox_batch(masks_per_animal, image_index, crop_ratio=1.5)

        assert set(result.keys()) == {"mouse_0", "mouse_1"}
        for df in result.values():
            assert list(df.columns) == ["x", "y", "h", "w"]
            assert len(df) == 5


# ---------------------------------------------------------------------------
# Tests: apply_mask_to_images
# ---------------------------------------------------------------------------


class TestApplyMaskToImages:

    def test_mask_removes_background(self, sample_masks, sample_data_dir):
        data_dir, _, image_paths = sample_data_dir
        output_dir = data_dir.parent / "masked"

        apply_mask_to_images(sample_masks, image_paths, data_dir, output_dir)

        from PIL import Image

        for img_path in image_paths:
            masked_img = np.array(Image.open(output_dir / img_path))
            assert masked_img[0, 0].sum() == 0, "Background pixel should be zeroed"
            mask_region = masked_img[20:50, 10:40]
            assert mask_region.sum() > 0, "Mask region should have non-zero pixels"


# ---------------------------------------------------------------------------
# Tests: prepare_animal_dataset
# ---------------------------------------------------------------------------


class TestPrepareAnimalDataset:

    def test_creates_output_structure(self, sample_masks, sample_data_dir):
        data_dir, csv_path, image_paths = sample_data_dir
        output_dir = data_dir.parent / "output"
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_paths, crop_ratio=1.5)

        result = prepare_animal_dataset(
            animal_id="mouse_0",
            bbox_df=bbox_df,
            input_data_dir=data_dir,
            input_csv_file=csv_path,
            output_base_dir=output_dir,
        )

        assert result["data_dir"].exists()
        assert result["csv_file"].exists()
        assert result["bbox_file"].exists()

    def test_bbox_csv_saved_correctly(self, sample_masks, sample_data_dir):
        data_dir, csv_path, image_paths = sample_data_dir
        output_dir = data_dir.parent / "output"
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_paths, crop_ratio=1.5)

        result = prepare_animal_dataset(
            animal_id="mouse_0",
            bbox_df=bbox_df,
            input_data_dir=data_dir,
            input_csv_file=csv_path,
            output_base_dir=output_dir,
        )

        saved_bbox = pd.read_csv(result["bbox_file"], index_col=0)
        assert list(saved_bbox.columns) == ["x", "y", "h", "w"]
        assert len(saved_bbox) == 5

    def test_with_mask_background_removal(self, sample_masks, sample_data_dir):
        data_dir, csv_path, image_paths = sample_data_dir
        output_dir = data_dir.parent / "output_masked"
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_paths, crop_ratio=1.5)

        result = prepare_animal_dataset(
            animal_id="mouse_0",
            bbox_df=bbox_df,
            input_data_dir=data_dir,
            input_csv_file=csv_path,
            output_base_dir=output_dir,
            masks=sample_masks,
            image_paths=image_paths,
        )

        assert result["data_dir"].exists()

    def test_masks_without_image_paths_raises(self, sample_masks, sample_data_dir):
        data_dir, csv_path, image_paths = sample_data_dir
        output_dir = data_dir.parent / "output"
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_paths, crop_ratio=1.5)

        with pytest.raises(ValueError, match="image_paths required"):
            prepare_animal_dataset(
                animal_id="mouse_0",
                bbox_df=bbox_df,
                input_data_dir=data_dir,
                input_csv_file=csv_path,
                output_base_dir=output_dir,
                masks=sample_masks,
            )

    def test_path_traversal_animal_id_sanitized(self, sample_masks, sample_data_dir):
        """Animal ID with path traversal chars should be sanitized."""
        data_dir, csv_path, image_paths = sample_data_dir
        output_dir = data_dir.parent / "output_sanitize"
        bbox_df = sam_masks_to_lp_bbox(sample_masks, image_paths, crop_ratio=1.5)

        result = prepare_animal_dataset(
            animal_id="../../etc/backdoor",
            bbox_df=bbox_df,
            input_data_dir=data_dir,
            input_csv_file=csv_path,
            output_base_dir=output_dir,
        )

        # The output dir should be within output_base_dir, not escaped
        assert output_dir in result["data_dir"].parents or result["data_dir"].is_relative_to(
            output_dir
        )
        # No ".." in the resolved path
        assert ".." not in str(result["data_dir"].resolve())


# ---------------------------------------------------------------------------
# Tests: merge_multi_animal_predictions
# ---------------------------------------------------------------------------


class TestMergeMultiAnimalPredictions:

    def test_merge_creates_output(self, tmp_path):
        header = pd.MultiIndex.from_tuples(
            [
                ("scorer", "bp1", "x"),
                ("scorer", "bp1", "y"),
                ("scorer", "bp2", "x"),
                ("scorer", "bp2", "y"),
            ]
        )
        for animal_id in ["mouse_0", "mouse_1"]:
            data = np.random.rand(10, 4) * 100
            df = pd.DataFrame(data, index=range(10), columns=header)
            df.to_csv(tmp_path / f"{animal_id}_preds.csv")

        pred_files = {
            "mouse_0": tmp_path / "mouse_0_preds.csv",
            "mouse_1": tmp_path / "mouse_1_preds.csv",
        }
        output_file = tmp_path / "merged.csv"
        merged = merge_multi_animal_predictions(pred_files, output_file)

        assert output_file.exists()
        assert len(merged) == 10
        assert merged.shape[1] == 8
        top_level = merged.columns.get_level_values(0).unique()
        assert set(top_level) == {"mouse_0", "mouse_1"}


# ---------------------------------------------------------------------------
# Tests: sam_eval — split_bodyparts_by_animal
# ---------------------------------------------------------------------------


class TestSplitBodypartsByAnimal:

    def test_two_animals_with_prefix(self):
        bodyparts = [
            "black_mouse_nose", "black_mouse_ear",
            "white_mouse_nose", "white_mouse_ear",
        ]
        result = split_bodyparts_by_animal(bodyparts)
        assert set(result.keys()) == {"black_mouse", "white_mouse"}
        assert len(result["black_mouse"]) == 2
        assert len(result["white_mouse"]) == 2

    def test_explicit_prefixes(self):
        bodyparts = [
            "black_mouse_nose", "black_mouse_ear",
            "white_mouse_nose", "white_mouse_ear",
        ]
        result = split_bodyparts_by_animal(bodyparts, animal_prefixes=["black_mouse"])
        assert set(result.keys()) == {"black_mouse"}
        assert len(result["black_mouse"]) == 2

    def test_no_prefix_single_animal_fallback(self):
        """Bodyparts without underscore-separated prefixes fall back to single animal."""
        bodyparts = ["nose", "ear", "tail"]
        result = split_bodyparts_by_animal(bodyparts)
        # Should not be empty — should fall back to treating as single animal
        assert len(result) == 1
        assert "animal" in result
        assert result["animal"] == bodyparts

    def test_numbered_animals(self):
        bodyparts = ["m1_nose", "m1_ear", "m2_nose", "m2_ear"]
        result = split_bodyparts_by_animal(bodyparts)
        assert set(result.keys()) == {"m1", "m2"}


# ---------------------------------------------------------------------------
# Tests: sam_eval — compute_euclidean_distances
# ---------------------------------------------------------------------------


class TestComputeEuclideanDistances:

    def _make_csv(self, tmp_path, name, header, data, index):
        df = pd.DataFrame(data, index=index, columns=header)
        path = tmp_path / name
        df.to_csv(path)
        return path

    def test_basic_distance(self, tmp_path):
        """Perfect predictions should have zero distance."""
        header = pd.MultiIndex.from_tuples([
            ("scorer", "animal_nose", "x"),
            ("scorer", "animal_nose", "y"),
        ])
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        gt_path = self._make_csv(tmp_path, "gt.csv", header, data, range(2))
        pred_path = self._make_csv(tmp_path, "pred.csv", header, data, range(2))

        result = compute_euclidean_distances(pred_path, gt_path, animal_prefixes=["animal"])
        assert "animal" in result
        assert np.allclose(result["animal"]["animal_nose"].values, 0.0)

    def test_merged_format_scopes_correctly(self, tmp_path):
        """Merged predictions with animal-level index should scope correctly."""
        # Ground truth with prefixed bodyparts
        gt_header = pd.MultiIndex.from_tuples([
            ("scorer", "mouse_0_nose", "x"),
            ("scorer", "mouse_0_nose", "y"),
            ("scorer", "mouse_1_nose", "x"),
            ("scorer", "mouse_1_nose", "y"),
        ])
        gt_data = np.array([[10.0, 20.0, 50.0, 60.0], [30.0, 40.0, 70.0, 80.0]])
        gt_path = self._make_csv(tmp_path, "gt.csv", gt_header, gt_data, range(2))

        # Merged predictions with animal_id at level 0
        pred_header = pd.MultiIndex.from_tuples([
            ("mouse_0", "nose", "x"),
            ("mouse_0", "nose", "y"),
            ("mouse_1", "nose", "x"),
            ("mouse_1", "nose", "y"),
        ])
        pred_data = np.array([[10.0, 20.0, 50.0, 60.0], [30.0, 40.0, 70.0, 80.0]])
        pred_path = self._make_csv(tmp_path, "pred.csv", pred_header, pred_data, range(2))

        result = compute_euclidean_distances(
            pred_path, gt_path, animal_prefixes=["mouse_0", "mouse_1"]
        )
        assert "mouse_0" in result
        assert "mouse_1" in result
        # Perfect match — distances should be 0
        assert np.allclose(result["mouse_0"]["mouse_0_nose"].values, 0.0)
        assert np.allclose(result["mouse_1"]["mouse_1_nose"].values, 0.0)


# ---------------------------------------------------------------------------
# Tests: sam_eval — summarize_distances
# ---------------------------------------------------------------------------


class TestSummarizeDistances:

    def test_summary_includes_all_rows(self):
        distances = {
            "animal_0": pd.DataFrame(
                {"nose": [1.0, 2.0, 3.0], "ear": [4.0, 5.0, 6.0]},
                index=range(3),
            )
        }
        summary = summarize_distances(distances)
        # Should have per-keypoint rows + 1 _ALL_ row
        assert len(summary) == 3  # nose, ear, _ALL_
        assert "_ALL_" in summary["keypoint"].values

    def test_no_dead_summary_variable(self):
        """Verify the summarize_distances function returns correct data
        (regression test for the dead-variable bug)."""
        distances = {
            "a": pd.DataFrame({"bp1": [1.0, 2.0]}, index=range(2))
        }
        result = summarize_distances(distances)
        # The _ALL_ row should be present in the result
        all_rows = result[result["keypoint"] == "_ALL_"]
        assert len(all_rows) == 1
        assert all_rows.iloc[0]["mean"] == pytest.approx(1.5)
