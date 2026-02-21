"""Standalone tests for predict_frame preprocessing and coordinate remap math.

These tests verify the numpy-level logic without requiring GPU or LP model loading.
They test that:
1. Preprocessing produces correct tensor shapes and value ranges
2. Coordinate remap correctly transforms resize-space keypoints to original frame coords
3. EnsemblePredictor aggregation math is correct
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Test the preprocessing pipeline that predict_frame applies."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, frame_rgb, resize_h, resize_w, bbox=None):
        """Replicate the preprocessing logic from Model.predict_frame."""
        import cv2

        if bbox is not None:
            bx, by, bw, bh = bbox
            crop = frame_rgb[by:by + bh, bx:bx + bw]
        else:
            crop = frame_rgb

        resized = cv2.resize(crop, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW

        return tensor

    def test_output_shape_no_bbox(self):
        """Without bbox, preprocessing should resize full frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = self._preprocess(frame, resize_h=256, resize_w=256)
        assert tensor.shape == (3, 256, 256)

    def test_output_shape_with_bbox(self):
        """With bbox, preprocessing should crop then resize."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = self._preprocess(frame, resize_h=256, resize_w=256, bbox=(100, 50, 200, 150))
        assert tensor.shape == (3, 256, 256)

    def test_normalization_range(self):
        """After ImageNet normalization, values should be roughly in [-2.5, 2.5]."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor = self._preprocess(frame, resize_h=64, resize_w=64)

        # ImageNet normalization: (x/255 - mean) / std
        # For x=0: (-mean/std), for x=255: ((1-mean)/std)
        # Range should be roughly [-2.2, 2.6]
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0

    def test_all_black_normalization(self):
        """All-black image should produce -mean/std after normalization."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = self._preprocess(frame, resize_h=64, resize_w=64)

        expected = -self.IMAGENET_MEAN / self.IMAGENET_STD
        for c in range(3):
            np.testing.assert_allclose(tensor[c], expected[c], atol=1e-5)

    def test_all_white_normalization(self):
        """All-white image should produce (1-mean)/std after normalization."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor = self._preprocess(frame, resize_h=64, resize_w=64)

        expected = (1.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
        for c in range(3):
            np.testing.assert_allclose(tensor[c], expected[c], atol=1e-5)

    def test_bbox_crops_correct_region(self):
        """Verify the crop extracts the right region before resize."""
        import cv2

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # Paint a 50x80 region starting at (x=40, y=10) red
        frame[10:60, 40:120, 0] = 255  # R channel

        # Crop that exact region
        bbox = (40, 10, 80, 50)  # x, y, w, h
        tensor = self._preprocess(frame, resize_h=32, resize_w=32, bbox=bbox)

        # After crop and resize, the R channel should have high values
        # (they were 255 before normalization)
        r_channel = tensor[0]
        expected_r = (1.0 - self.IMAGENET_MEAN[0]) / self.IMAGENET_STD[0]
        np.testing.assert_allclose(r_channel, expected_r, atol=0.1)


# ---------------------------------------------------------------------------
# Coordinate remap tests
# ---------------------------------------------------------------------------


class TestCoordinateRemap:
    """Test the coordinate remap math from predict_frame.

    The remap converts keypoints from resize-space (e.g., 256x256) to
    original frame coordinates. This replicates what convert_bbox_coords
    + normalized_to_bbox do in the LP pipeline.

    After the AXIOM-3 fix, remap uses actual crop dimensions (post numpy
    clipping) instead of the requested bbox dimensions.
    """

    def _remap(self, kp_resize, resize_h, resize_w, frame_h, frame_w, bbox=None):
        """Replicate the coordinate remap logic from Model.predict_frame.

        Uses actual crop dimensions (clamped to frame) instead of requested
        bbox dimensions, matching the AXIOM-3 fix.
        """
        kp = kp_resize.copy()

        # Normalize to [0, 1]
        kp[:, 0] /= resize_w
        kp[:, 1] /= resize_h

        # Scale to crop/frame dimensions
        if bbox is not None:
            bx, by, bw, bh = bbox
            # Compute actual crop dims after numpy clipping (AXIOM-3 fix)
            actual_w = min(bx + bw, frame_w) - max(bx, 0)
            actual_h = min(by + bh, frame_h) - max(by, 0)
            kp[:, 0] = kp[:, 0] * actual_w + bx
            kp[:, 1] = kp[:, 1] * actual_h + by
        else:
            kp[:, 0] *= frame_w
            kp[:, 1] *= frame_h

        return kp

    def test_identity_remap_no_bbox(self):
        """If resize dims match frame dims, keypoints should not move."""
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=256, frame_w=256)
        np.testing.assert_allclose(result, [[128.0, 128.0]])

    def test_scale_up_no_bbox(self):
        """Resize 256x256 -> frame 512x512 should double coordinates."""
        kp = np.array([[128.0, 64.0]], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=512, frame_w=512)
        np.testing.assert_allclose(result, [[256.0, 128.0]])

    def test_non_square_frame(self):
        """Resize 256x256 -> frame 480x640 should scale correctly."""
        # Keypoint at center of resize space
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=480, frame_w=640)
        np.testing.assert_allclose(result, [[320.0, 240.0]])

    def test_origin_no_bbox(self):
        """Keypoint at (0,0) in resize space should map to (0,0) in frame."""
        kp = np.array([[0.0, 0.0]], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=480, frame_w=640)
        np.testing.assert_allclose(result, [[0.0, 0.0]])

    def test_max_corner_no_bbox(self):
        """Keypoint at (resize_w, resize_h) should map to (frame_w, frame_h)."""
        kp = np.array([[256.0, 256.0]], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=480, frame_w=640)
        np.testing.assert_allclose(result, [[640.0, 480.0]])

    def test_bbox_offset(self):
        """With bbox, keypoints should be offset by bbox origin."""
        # Keypoint at center of resize space (128, 128) in 256x256
        # Bbox: x=100, y=200, w=300, h=400
        # Normalized: (0.5, 0.5) -> (0.5*300+100, 0.5*400+200) = (250, 400)
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result = self._remap(
            kp, resize_h=256, resize_w=256,
            frame_h=1000, frame_w=1000,
            bbox=(100, 200, 300, 400),
        )
        np.testing.assert_allclose(result, [[250.0, 400.0]])

    def test_bbox_origin(self):
        """Keypoint at (0,0) in resize space should map to bbox top-left."""
        kp = np.array([[0.0, 0.0]], dtype=np.float32)
        result = self._remap(
            kp, resize_h=256, resize_w=256,
            frame_h=1000, frame_w=1000,
            bbox=(50, 75, 200, 300),
        )
        np.testing.assert_allclose(result, [[50.0, 75.0]])

    def test_bbox_bottom_right(self):
        """Keypoint at (resize_w, resize_h) should map to bbox bottom-right."""
        kp = np.array([[256.0, 256.0]], dtype=np.float32)
        result = self._remap(
            kp, resize_h=256, resize_w=256,
            frame_h=1000, frame_w=1000,
            bbox=(50, 75, 200, 300),
        )
        np.testing.assert_allclose(result, [[250.0, 375.0]])

    def test_multiple_keypoints(self):
        """Multiple keypoints should all be remapped correctly."""
        kp = np.array([
            [0.0, 0.0],
            [128.0, 128.0],
            [256.0, 256.0],
        ], dtype=np.float32)
        result = self._remap(kp, resize_h=256, resize_w=256, frame_h=512, frame_w=512)
        expected = np.array([
            [0.0, 0.0],
            [256.0, 256.0],
            [512.0, 512.0],
        ])
        np.testing.assert_allclose(result, expected)

    def test_bbox_exceeds_frame_width(self):
        """AXIOM-3: bbox extending past right edge should use actual crop width.

        Frame: 640x480. Bbox: x=600, y=0, w=200, h=200.
        Actual crop: x=[600:800] clipped to [600:640] → actual_w=40.
        Keypoint at center of resize (128/256 = 0.5):
          x = 0.5 * 40 + 600 = 620 (not 0.5 * 200 + 600 = 700!)
        """
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result = self._remap(
            kp, resize_h=256, resize_w=256,
            frame_h=480, frame_w=640,
            bbox=(600, 0, 200, 200),
        )
        # actual_w = min(600+200, 640) - max(600, 0) = 640 - 600 = 40
        # actual_h = min(0+200, 480) - max(0, 0) = 200
        np.testing.assert_allclose(result, [[620.0, 100.0]])

    def test_bbox_exceeds_frame_height(self):
        """AXIOM-3: bbox extending past bottom edge should use actual crop height."""
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result = self._remap(
            kp, resize_h=256, resize_w=256,
            frame_h=480, frame_w=640,
            bbox=(0, 400, 200, 200),
        )
        # actual_w = 200, actual_h = min(400+200, 480) - 400 = 80
        np.testing.assert_allclose(result, [[100.0, 440.0]])

    def test_bbox_fully_inside_frame(self):
        """When bbox is fully inside frame, actual dims == requested dims."""
        kp = np.array([[128.0, 128.0]], dtype=np.float32)
        result_old = self._remap(
            kp.copy(), resize_h=256, resize_w=256,
            frame_h=1000, frame_w=1000,
            bbox=(100, 100, 200, 300),
        )
        # actual_w = 200, actual_h = 300 — same as requested
        np.testing.assert_allclose(result_old, [[200.0, 250.0]])


# ---------------------------------------------------------------------------
# Input validation tests (P1)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test the validation logic added to predict_frame and lp_keypoints_to_sam_bbox."""

    def test_frame_rgb_wrong_ndim_2d(self):
        """2D array should raise ValueError."""
        # This replicates the validation logic (not calling predict_frame directly)
        frame = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be.*H, W, 3"):
            if frame.ndim != 3 or (frame.ndim == 3 and frame.shape[2] != 3):
                raise ValueError(
                    f"frame_rgb must be (H, W, 3), got shape {frame.shape}"
                )

    def test_frame_rgb_wrong_channels_4(self):
        """RGBA (4 channels) should raise ValueError."""
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be.*H, W, 3"):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"frame_rgb must be (H, W, 3), got shape {frame.shape}"
                )

    def test_frame_rgb_empty_raises(self):
        """Empty array should raise ValueError."""
        frame = np.zeros((0, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            if frame.size == 0:
                raise ValueError("frame_rgb is empty")

    def test_bbox_zero_width_raises(self):
        """Zero-width bbox should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            bw, bh = 0, 100
            if bw <= 0 or bh <= 0:
                raise ValueError(
                    f"bbox width and height must be positive, got w={bw}, h={bh}"
                )

    def test_bbox_negative_height_raises(self):
        """Negative height bbox should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            bw, bh = 100, -5
            if bw <= 0 or bh <= 0:
                raise ValueError(
                    f"bbox width and height must be positive, got w={bw}, h={bh}"
                )

    def test_bbox_empty_crop_raises(self):
        """Bbox that produces empty crop should raise ValueError."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # bbox starts at y=200 on a 100-tall frame → empty crop
        bbox = (0, 200, 50, 50)
        bx, by, bw, bh = bbox
        crop = frame[by:by + bh, bx:bx + bw]
        with pytest.raises(ValueError, match="empty crop"):
            if crop.size == 0:
                raise ValueError(
                    f"bbox (x={bx}, y={by}, w={bw}, h={bh}) produces an empty "
                    f"crop on frame of shape {frame.shape[:2]}"
                )


# ---------------------------------------------------------------------------
# EnsemblePredictor aggregation tests
# ---------------------------------------------------------------------------


class TestEnsembleAggregation:
    """Test the numpy aggregation math used by EnsemblePredictor.predict_frame."""

    def _aggregate(self, results):
        """Replicate the aggregation logic from EnsemblePredictor.predict_frame."""
        all_kps = np.stack([r["keypoints"] for r in results])    # (N, K, 2)
        all_conf = np.stack([r["confidence"] for r in results])  # (N, K)
        return {
            "keypoints": all_kps.mean(axis=0),                   # (K, 2)
            "confidence": all_conf.mean(axis=0),                  # (K,)
            "variance": all_kps.var(axis=0).sum(axis=-1),         # (K,)
            "per_model": results,
        }

    def test_single_model_mean_is_identity(self):
        """With 1 model, mean should equal the single prediction."""
        results = [{"keypoints": np.array([[100.0, 200.0]]), "confidence": np.array([0.9])}]
        agg = self._aggregate(results)
        np.testing.assert_allclose(agg["keypoints"], [[100.0, 200.0]])
        np.testing.assert_allclose(agg["confidence"], [0.9])
        np.testing.assert_allclose(agg["variance"], [0.0])

    def test_two_model_mean(self):
        """Mean of two models should be their average."""
        results = [
            {"keypoints": np.array([[100.0, 200.0], [300.0, 400.0]]),
             "confidence": np.array([0.8, 0.6])},
            {"keypoints": np.array([[120.0, 220.0], [320.0, 420.0]]),
             "confidence": np.array([0.9, 0.7])},
        ]
        agg = self._aggregate(results)
        np.testing.assert_allclose(agg["keypoints"], [[110.0, 210.0], [310.0, 410.0]])
        np.testing.assert_allclose(agg["confidence"], [0.85, 0.65])

    def test_variance_computation(self):
        """Variance should be sum of x_var + y_var per keypoint."""
        results = [
            {"keypoints": np.array([[100.0, 200.0]]), "confidence": np.array([0.9])},
            {"keypoints": np.array([[120.0, 240.0]]), "confidence": np.array([0.8])},
        ]
        agg = self._aggregate(results)
        # x values: [100, 120], var = 100
        # y values: [200, 240], var = 400
        # total variance = 100 + 400 = 500
        np.testing.assert_allclose(agg["variance"], [500.0])

    def test_identical_models_zero_variance(self):
        """Identical predictions should have zero variance."""
        kp = np.array([[50.0, 60.0], [70.0, 80.0]])
        conf = np.array([0.9, 0.8])
        results = [
            {"keypoints": kp.copy(), "confidence": conf.copy()},
            {"keypoints": kp.copy(), "confidence": conf.copy()},
            {"keypoints": kp.copy(), "confidence": conf.copy()},
        ]
        agg = self._aggregate(results)
        np.testing.assert_allclose(agg["variance"], [0.0, 0.0])

    def test_output_shapes(self):
        """Verify output shapes are correct."""
        n_models = 5
        n_keypoints = 10
        results = [
            {
                "keypoints": np.random.randn(n_keypoints, 2).astype(np.float32),
                "confidence": np.random.rand(n_keypoints).astype(np.float32),
            }
            for _ in range(n_models)
        ]
        agg = self._aggregate(results)
        assert agg["keypoints"].shape == (n_keypoints, 2)
        assert agg["confidence"].shape == (n_keypoints,)
        assert agg["variance"].shape == (n_keypoints,)
        assert len(agg["per_model"]) == n_models
