"""Standalone tests for lightning_pose.utils.qc — numpy/pandas only, no LP deps."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import qc.py directly, bypassing lightning_pose.__init__ which requires
# omegaconf.  This allows tests to run in environments without the full LP
# dependency stack.
# ---------------------------------------------------------------------------

_utils_path = Path(__file__).resolve().parent.parent.parent / "lightning_pose" / "utils"

# Placeholder parent modules so importlib can resolve the package chain
_fake_lp = mock.MagicMock()
_fake_lp_utils = mock.MagicMock()
sys.modules.setdefault("lightning_pose", _fake_lp)
sys.modules.setdefault("lightning_pose.utils", _fake_lp_utils)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _utils_path / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_qc_mod = _load_module("lightning_pose.utils.qc", "qc.py")

# Public API
QCResult = _qc_mod.QCResult
flag_outlier_frames = _qc_mod.flag_outlier_frames
format_qc_summary = _qc_mod.format_qc_summary
save_qc_report = _qc_mod.save_qc_report
_normalize_ensemble_variance = _qc_mod._normalize_ensemble_variance
HMMDetector = _qc_mod.HMMDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temporal_norm(n_frames: int, keypoints: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = rng.exponential(scale=2.0, size=(n_frames, len(keypoints)))
    return pd.DataFrame(data, columns=keypoints)


def _make_ensemble_variance(
    n_frames: int, keypoints: list[str], seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cols = []
    data_arrays = []
    for kp in keypoints:
        x_var = rng.exponential(scale=1.0, size=n_frames)
        y_var = rng.exponential(scale=1.0, size=n_frames)
        total_var = x_var + y_var
        cols.extend([
            ("ensemble", kp, "x_var"),
            ("ensemble", kp, "y_var"),
            ("ensemble", kp, "total_var"),
        ])
        data_arrays.extend([x_var, y_var, total_var])
    mi = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    return pd.DataFrame(np.column_stack(data_arrays), columns=mi)


def _make_bimodal_metric(
    n_frames: int, keypoints: list[str], error_frames: dict[str, list[int]],
    clean_mean: float = 2.0, error_mean: float = 50.0, seed: int = 0,
) -> pd.DataFrame:
    """Create a metric DataFrame with clear clean/error regimes."""
    rng = np.random.RandomState(seed)
    data = rng.exponential(scale=clean_mean, size=(n_frames, len(keypoints)))
    df = pd.DataFrame(data, columns=keypoints)
    for kp, frames in error_frames.items():
        df.loc[frames, kp] = rng.normal(error_mean, 5.0, len(frames))
    return df


# ---------------------------------------------------------------------------
# flag_outlier_frames (API-level tests, detector-agnostic)
# ---------------------------------------------------------------------------

class TestFlagOutlierFrames:
    def test_no_metrics_raises(self):
        with pytest.raises(ValueError, match="At least one metric"):
            flag_outlier_frames()

    def test_returns_qc_result(self):
        df = _make_temporal_norm(200, ["nose", "ear"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df)
        assert isinstance(result, QCResult)
        assert result.flags_df.shape[1] == 2
        assert result.posteriors_df is not None
        assert result.detector_name == "hmm"

    def test_error_block_detected(self):
        """A block of anomalous frames should be flagged."""
        n = 500
        kps = ["nose", "ear"]
        error_frames = {"nose": list(range(245, 260))}
        df = _make_bimodal_metric(n, kps, error_frames, error_mean=50.0)

        result = flag_outlier_frames(temporal_norm_df=df)
        flagged = result.flags_df.loc[245:259, "nose"].sum()
        assert flagged >= 5, f"Only {flagged}/15 error frames flagged"
        assert result.summary["n_flagged_frames"] >= 5
        assert result.summary["per_keypoint"]["nose"] >= 5

    def test_ensemble_variance_multiindex(self):
        n = 500
        kps = ["nose", "tail"]
        df = _make_ensemble_variance(n, kps, seed=10)
        df.loc[250, ("ensemble", "nose", "total_var")] = 5000.0

        result = flag_outlier_frames(ensemble_variance_df=df)
        assert "nose" in result.flags_df.columns
        assert "tail" in result.flags_df.columns

    def test_summary_structure(self):
        df = _make_temporal_norm(200, ["nose"], seed=7)
        result = flag_outlier_frames(temporal_norm_df=df)
        s = result.summary
        assert "n_frames" in s
        assert "n_flagged_frames" in s
        assert "pct_flagged_frames" in s
        assert "per_keypoint" in s
        assert "nose" in s["per_keypoint"]

    def test_custom_detector(self):
        """A custom detector implementing fit_score should work."""

        class AlwaysFlagDetector:
            name = "always_flag"

            def fit_score(self, metrics, n_frames, keypoints):
                return pd.DataFrame(1.0, index=range(n_frames), columns=keypoints)

        df = _make_temporal_norm(50, ["a"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df, detector=AlwaysFlagDetector())
        assert result.flags_df["a"].all()
        assert result.detector_name == "always_flag"

    def test_custom_threshold(self):
        """Setting threshold=0.9 should flag fewer frames than 0.1."""
        n = 500
        df = _make_bimodal_metric(n, ["nose"], error_frames={"nose": list(range(450, 500))})

        result_low = flag_outlier_frames(temporal_norm_df=df, threshold=0.1)
        result_high = flag_outlier_frames(temporal_norm_df=df, threshold=0.9)
        assert result_low.summary["n_flagged_frames"] >= result_high.summary["n_flagged_frames"]

    def test_threshold_nan_raises(self):
        """NaN threshold silently produces zero flags — must reject."""
        df = _make_temporal_norm(200, ["nose"], seed=1)
        with pytest.raises(ValueError, match="threshold"):
            flag_outlier_frames(temporal_norm_df=df, threshold=float("nan"))

    def test_threshold_out_of_range_raises(self):
        df = _make_temporal_norm(200, ["nose"], seed=1)
        with pytest.raises(ValueError, match="threshold"):
            flag_outlier_frames(temporal_norm_df=df, threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            flag_outlier_frames(temporal_norm_df=df, threshold=1.0)
        with pytest.raises(ValueError, match="threshold"):
            flag_outlier_frames(temporal_norm_df=df, threshold=-0.5)

    def test_nan_posteriors_flagged_not_clean(self):
        """NaN > threshold = False in numpy — NaN posteriors must be flagged."""

        class NaNDetector:
            name = "nan_bug"

            def fit_score(self, metrics, n_frames, keypoints):
                df = pd.DataFrame(0.0, index=range(n_frames), columns=keypoints)
                # Simulate numeric bug: some posteriors are NaN
                df.loc[5, "nose"] = float("nan")
                df.loc[10, "nose"] = float("nan")
                return df

        df = _make_temporal_norm(200, ["nose", "ear"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df, detector=NaNDetector())
        # NaN frames must be flagged (True), not clean (False)
        assert result.flags_df.loc[5, "nose"] == True
        assert result.flags_df.loc[10, "nose"] == True
        # Non-NaN frames at 0.0 should remain clean
        assert result.flags_df.loc[0, "nose"] == False

    def test_wrong_shape_detector_raises(self):
        """Detector returning wrong-shape DataFrame must raise ValueError."""

        class WrongShapeDetector:
            name = "wrong"

            def fit_score(self, metrics, n_frames, keypoints):
                # Return wrong number of rows
                return pd.DataFrame(0.0, index=range(n_frames + 10), columns=keypoints)

        df = _make_temporal_norm(200, ["nose"], seed=1)
        with pytest.raises(ValueError, match="shape"):
            flag_outlier_frames(temporal_norm_df=df, detector=WrongShapeDetector())


# ---------------------------------------------------------------------------
# HMMDetector
# ---------------------------------------------------------------------------

class TestHMMDetector:
    def test_sustained_error_detection(self):
        """HMM should detect a sustained level shift (simulating an ID swap)."""
        n = 1000
        kps = ["nose"]
        # Frames 500-550: sustained high values (ID swap)
        error_frames = {"nose": list(range(500, 550))}
        df = _make_bimodal_metric(n, kps, error_frames, clean_mean=2.0, error_mean=40.0)

        det = HMMDetector(expected_error_duration=10.0)
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)

        # Most of the error frames should be flagged
        flagged_in_error = result.flags_df.loc[500:549, "nose"].sum()
        assert flagged_in_error >= 30, f"Only {flagged_in_error}/50 error frames flagged"

    def test_clean_data_few_flags(self):
        """Clean unimodal data should produce very few flags."""
        n = 1000
        df = _make_temporal_norm(n, ["a", "b"], seed=99)
        result = flag_outlier_frames(temporal_norm_df=df)
        # HMM with BIC/pi_1 checks should reject spurious bimodality.
        # Allow some flags from quantile fallback or borderline cases.
        assert result.summary["pct_flagged_frames"] < 15.0

    def test_all_nan_column_flagged(self):
        """All-NaN metric = no model could track → should be flagged, not clean."""
        n = 200
        df = pd.DataFrame({
            "nose": np.random.exponential(2.0, n),
            "tail": np.full(n, np.nan),
        })
        det = HMMDetector()
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        # All-NaN keypoint should be fully flagged (posterior=1.0)
        assert result.posteriors_df["tail"].iloc[0] == 1.0
        assert result.flags_df["tail"].all()
        # "nose" should still be scored normally (not all flagged)
        assert not result.flags_df["nose"].all()

    def test_inf_in_metrics_does_not_corrupt(self):
        """Inf values must not produce NaN posteriors (AXIOM-1, AEGIS BUG-1).

        Before fix: np.log(Inf) → Inf → _log_normal_pdf returns -Inf →
        logaddexp(-Inf, -Inf) → NaN → corrupts entire forward-backward pass.
        """
        n = 500
        df = _make_temporal_norm(n, ["nose", "ear"], seed=42)
        # Inject Inf at several frames
        df.loc[10, "nose"] = np.inf
        df.loc[20, "nose"] = np.inf
        df.loc[30, "ear"] = np.inf

        result = flag_outlier_frames(temporal_norm_df=df)
        # No NaN in posteriors
        assert not result.posteriors_df.isna().any().any(), (
            "Inf in metrics produced NaN posteriors"
        )
        # Flags should still be boolean (no NaN)
        assert result.flags_df.dtypes.apply(lambda d: d == bool).all()

    def test_short_sequence_uses_fallback(self):
        """Sequences shorter than min_frames_for_hmm use quantile fallback."""
        n = 50  # below default min_frames_for_hmm=100
        df = _make_temporal_norm(n, ["nose"], seed=5)
        df.loc[0, "nose"] = 9999.0

        det = HMMDetector()
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        # The spike should still be flagged via quantile fallback
        assert result.posteriors_df.loc[0, "nose"] > 0.5

    def test_posteriors_have_range(self):
        """Posteriors should be high for error frames and low for clean frames."""
        n = 500
        kps = ["nose"]
        error_frames = {"nose": list(range(200, 220))}
        df = _make_bimodal_metric(n, kps, error_frames,
                                  clean_mean=2.0, error_mean=50.0, seed=42)

        det = HMMDetector()
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        assert result.posteriors_df is not None
        assert result.posteriors_df["nose"].max() > 0.5
        assert result.posteriors_df["nose"].min() < 0.5

    def test_multiple_metrics_combined(self):
        """Multiple metrics should be combined via independent per-metric HMMs."""
        n = 1000
        kps = ["nose"]
        # Create clean baselines
        tn = _make_temporal_norm(n, kps, seed=1)
        ev = _make_ensemble_variance(n, kps, seed=2)

        # Block of error frames in temporal_norm at frames 100-115
        for i in range(100, 115):
            tn.loc[i, "nose"] = 5000.0
        # Block of error frames in ensemble_variance at frames 300-315
        for i in range(300, 315):
            ev.loc[i, ("ensemble", "nose", "total_var")] = 5000.0

        result = flag_outlier_frames(temporal_norm_df=tn, ensemble_variance_df=ev)
        # Both error blocks should be detected
        flagged_tn = result.flags_df.loc[100:114, "nose"].sum()
        flagged_ev = result.flags_df.loc[300:314, "nose"].sum()
        assert flagged_tn >= 5, f"Only {flagged_tn}/15 temporal_norm errors flagged"
        assert flagged_ev >= 5, f"Only {flagged_ev}/15 ensemble_var errors flagged"

    def test_hmm_outperforms_quantile_on_temporal_structure(self):
        """HMM should leverage temporal structure: catch contiguous blocks
        while suppressing isolated spikes that quantile flags as false positives.

        This justifies the HMM's additional complexity over simple
        percentile-based approaches. The HMM's transition probabilities
        reinforce contiguous anomalies and dampen isolated spikes.
        """
        n = 1000
        rng = np.random.RandomState(42)
        clean = rng.exponential(scale=1.0, size=n)
        df = pd.DataFrame({"nose": clean})

        # Contiguous error block: frames 400-449
        for i in range(400, 450):
            df.loc[i, "nose"] = rng.normal(15.0, 2.0)

        # Isolated spikes of same amplitude (not real errors, just noise)
        spike_frames = [50, 150, 250, 600, 700, 800, 900]
        for i in spike_frames:
            df.loc[i, "nose"] = rng.normal(15.0, 2.0)

        det = HMMDetector()

        # HMM path: full inference with temporal structure
        hmm_scores = det.fit_score(
            {"temporal_norm": df}, n_frames=n, keypoints=["nose"],
        )

        # Quantile fallback path: no temporal structure
        vals = df["nose"].values.astype(float)
        quantile_scores = det._quantile_fallback(
            {"temporal_norm": vals}, n_frames=n,
        )

        # HMM should catch the contiguous block
        hmm_block = int((hmm_scores.loc[400:449, "nose"] > 0.5).sum())
        assert hmm_block >= 40, (
            f"HMM only flagged {hmm_block}/50 contiguous error frames"
        )

        # HMM should suppress isolated spikes (temporal smoothing)
        hmm_spikes = sum(
            int(hmm_scores.loc[f, "nose"] > 0.5) for f in spike_frames
        )

        # Quantile flags spikes indiscriminately (no temporal context)
        q_spikes = sum(int(quantile_scores[f] > 0.5) for f in spike_frames)

        assert hmm_spikes < q_spikes, (
            f"HMM should suppress more isolated spikes than quantile: "
            f"HMM flagged {hmm_spikes}, quantile flagged {q_spikes}"
        )


# ---------------------------------------------------------------------------
# save_qc_report
# ---------------------------------------------------------------------------

class TestSaveQcReport:
    def test_files_created(self):
        df = _make_temporal_norm(200, ["nose", "ear"], seed=5)
        result = flag_outlier_frames(temporal_norm_df=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            flags_path = save_qc_report(result, tmpdir)
            assert flags_path.exists()
            assert (Path(tmpdir) / "qc_summary.csv").exists()
            assert (Path(tmpdir) / "qc_posteriors.csv").exists()

    def test_files_created_with_prefix(self):
        df = _make_temporal_norm(200, ["nose"], seed=6)
        result = flag_outlier_frames(temporal_norm_df=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            flags_path = save_qc_report(result, tmpdir, prefix="animal_0_")
            assert flags_path.name == "animal_0_qc_flags.csv"
            assert (Path(tmpdir) / "animal_0_qc_summary.csv").exists()
            assert (Path(tmpdir) / "animal_0_qc_posteriors.csv").exists()

    def test_round_trip_flags(self):
        n = 200
        df = _make_bimodal_metric(
            n, ["nose", "ear"],
            error_frames={"nose": [100], "ear": [150]},
            error_mean=100.0,
        )
        result = flag_outlier_frames(temporal_norm_df=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            flags_path = save_qc_report(result, tmpdir)
            loaded = pd.read_csv(flags_path, index_col=0)
            assert loaded.shape == result.flags_df.shape

    def test_summary_csv_includes_params(self):
        """qc_summary.csv must record detector and threshold for reproducibility."""
        df = _make_temporal_norm(200, ["nose", "ear"], seed=5)
        result = flag_outlier_frames(temporal_norm_df=df, threshold=0.4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qc_report(result, tmpdir)
            summary = pd.read_csv(Path(tmpdir) / "qc_summary.csv")
            assert "detector" in summary.columns
            assert "threshold" in summary.columns
            assert summary["detector"].iloc[0] == "hmm"
            assert summary["threshold"].iloc[0] == 0.4


# ---------------------------------------------------------------------------
# format_qc_summary
# ---------------------------------------------------------------------------

class TestFormatQcSummary:
    def test_output_contains_counts(self):
        df = _make_temporal_norm(500, ["nose", "left_ear"], seed=11)
        df.loc[10, "nose"] = 9999.0
        result = flag_outlier_frames(temporal_norm_df=df)

        text = format_qc_summary(result)
        assert "QC" in text
        assert "/500" in text
        assert "nose" in text
        assert "left_ear" in text
        assert "%" in text

    def test_detector_name_in_output(self):
        class DummyDetector:
            name = "dummy"
            def fit_score(self, metrics, n_frames, keypoints):
                return pd.DataFrame(0.0, index=range(n_frames), columns=keypoints)

        df = _make_temporal_norm(200, ["a"], seed=12)
        result = flag_outlier_frames(temporal_norm_df=df, detector=DummyDetector())
        text = format_qc_summary(result)
        assert "[dummy]" in text

    def test_zero_flags_message(self):
        """When no frames are flagged, show a clear 'no errors' message."""
        class NeverFlagDetector:
            name = "never"
            def fit_score(self, metrics, n_frames, keypoints):
                return pd.DataFrame(0.0, index=range(n_frames), columns=keypoints)

        df = _make_temporal_norm(200, ["nose"], seed=14)
        result = flag_outlier_frames(temporal_norm_df=df, detector=NeverFlagDetector())
        text = format_qc_summary(result)
        assert "No suspected tracking errors" in text

    def test_high_flag_rate_warning(self):
        """Keypoints flagged >30% should trigger a warning."""
        class AlwaysFlagDetector:
            name = "always"
            def fit_score(self, metrics, n_frames, keypoints):
                return pd.DataFrame(1.0, index=range(n_frames), columns=keypoints)

        df = _make_temporal_norm(100, ["nose", "ear"], seed=15)
        result = flag_outlier_frames(temporal_norm_df=df, detector=AlwaysFlagDetector())
        text = format_qc_summary(result)
        assert "WARNING" in text
        assert ">30%" in text

    def test_alignment(self):
        kps = ["a", "long_name"]
        df = _make_temporal_norm(200, kps, seed=13)
        result = flag_outlier_frames(temporal_norm_df=df)
        text = format_qc_summary(result)
        lines = text.strip().split("\n")
        assert len(lines) >= 3


# ---------------------------------------------------------------------------
# _normalize_ensemble_variance
# ---------------------------------------------------------------------------

class TestNormalizeEnsembleVariance:
    def test_multiindex_extraction(self):
        kps = ["nose", "tail"]
        df = _make_ensemble_variance(10, kps, seed=20)
        result = _normalize_ensemble_variance(df)
        assert list(result.columns) == kps
        assert result.shape == (10, 2)

    def test_flat_columns_passthrough(self):
        df = pd.DataFrame({"nose": [1.0, 2.0], "tail": [3.0, 4.0]})
        result = _normalize_ensemble_variance(df)
        assert list(result.columns) == ["nose", "tail"]


# ---------------------------------------------------------------------------
# CSV round-trip integration tests
# ---------------------------------------------------------------------------

class TestCSVRoundTrip:
    """Test QC against CSVs saved/loaded the way LP actually does it.

    LP metric CSVs use bare ``to_csv()`` with default params.
    temporal_norm and pca_singleview are flat (single header row);
    ensemble_variance has a 3-level MultiIndex header.
    """

    def test_flat_metric_csv_round_trip(self):
        """Flat temporal_norm CSV: save with to_csv(), read with index_col=0."""
        n = 500
        kps = ["nose", "ear"]
        df = _make_bimodal_metric(
            n, kps, error_frames={"nose": list(range(200, 215))},
            error_mean=50.0,
        )
        # First row NaN — mimics real temporal_norm (no previous frame)
        df.loc[0] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_temporal_norm.csv"
            # LP saves with bare to_csv()
            df.to_csv(csv_path)
            # Read back the way sam_detect.py should (flat, index_col=0)
            loaded = pd.read_csv(csv_path, index_col=0)

            result = flag_outlier_frames(temporal_norm_df=loaded)
            assert result.flags_df.shape == (n, 2)
            # Error block should be partially flagged
            flagged = result.flags_df.loc[200:214, "nose"].sum()
            assert flagged >= 3, f"Only {flagged}/15 error frames flagged"

    def test_multiindex_ensemble_csv_round_trip(self):
        """Ensemble variance CSV: save with to_csv(), read with header=[0,1,2]."""
        n = 300
        kps = ["nose", "tail"]
        df = _make_ensemble_variance(n, kps, seed=55)
        # Inject a spike
        df.loc[150, ("ensemble", "nose", "total_var")] = 5000.0

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ensemble_variance.csv"
            df.to_csv(csv_path)
            # Read back the way both predict.py and sam_detect.py do
            loaded = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

            result = flag_outlier_frames(ensemble_variance_df=loaded)
            assert "nose" in result.flags_df.columns
            assert "tail" in result.flags_df.columns

    def test_flat_csv_with_set_column(self):
        """Metric CSV with a 'set' column (from labeled data) should work."""
        n = 200
        df = _make_temporal_norm(n, ["nose", "ear"], seed=44)
        df["set"] = "train"
        df.loc[n // 2:, "set"] = "val"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_with_set.csv"
            df.to_csv(csv_path)
            loaded = pd.read_csv(csv_path, index_col=0)

            result = flag_outlier_frames(temporal_norm_df=loaded)
            # 'set' column should be dropped (non-numeric)
            assert "set" not in result.flags_df.columns
            assert result.flags_df.shape[1] == 2

    def test_zero_values_not_dropped(self):
        """Frames with value=0.0 should not be silently dropped."""
        n = 200
        df = _make_temporal_norm(n, ["nose"], seed=66)
        # Frame 0 is typically 0 in temporal_norm; add a few more
        df.loc[0, "nose"] = 0.0
        df.loc[1, "nose"] = 0.0
        df.loc[2, "nose"] = 0.0

        result = flag_outlier_frames(temporal_norm_df=df)
        # Zero frames should get low posteriors (clean), not NaN
        assert not np.isnan(result.posteriors_df.loc[0, "nose"])
        assert not np.isnan(result.posteriors_df.loc[1, "nose"])
        assert not np.isnan(result.posteriors_df.loc[2, "nose"])


# ---------------------------------------------------------------------------
# EM return value bug fix verification
# ---------------------------------------------------------------------------

class TestEMReturnFix:
    """Verify _fit_gaussian_mixture_em returns best_params, not stale loop vars.

    Addresses: SENTINEL-01, ENTROPY-8/14, VOSS-5, AEGIS BUG-1.
    """

    def test_em_returns_best_params_not_last(self):
        """Return value must match the best (highest LL) restart's canonicalized params."""
        det = HMMDetector()
        # Create clear bimodal data: most values low, some high
        rng = np.random.RandomState(123)
        clean = rng.normal(1.0, 0.3, 900)
        error = rng.normal(5.0, 0.3, 100)
        data = np.concatenate([clean, error])
        rng.shuffle(data)

        params = det._fit_gaussian_mixture_em(data)
        if params is None:
            pytest.skip("EM returned None (BIC preferred 1 component)")

        mu_0, sig_0, mu_1, sig_1, pi_1 = params
        # Canonical: component 0 = clean (lower mean), component 1 = error (higher mean)
        assert mu_0 < mu_1, (
            f"Canonicalization failed: mu_0={mu_0:.2f} >= mu_1={mu_1:.2f}. "
            f"This was the core bug — returning un-canonicalized params."
        )
        # Error fraction should be roughly 10%
        assert 0.01 < pi_1 < 0.35, f"pi_1={pi_1:.3f} outside expected range"

    def test_em_single_restart_still_works(self):
        """With _N_RESTARTS at default, params are correctly returned."""
        det = HMMDetector()
        rng = np.random.RandomState(42)
        clean = rng.normal(0.5, 0.2, 800)
        error = rng.normal(4.0, 0.3, 200)
        data = np.concatenate([clean, error])

        params = det._fit_gaussian_mixture_em(data)
        if params is not None:
            mu_0, _, mu_1, _, _ = params
            assert mu_0 < mu_1, "Canonicalization must hold even with 1 restart"


# ---------------------------------------------------------------------------
# HMMDetector parameter validation
# ---------------------------------------------------------------------------

class TestHMMDetectorValidation:
    """Verify HMMDetector rejects invalid constructor parameters.

    Addresses: AEGIS 2.6 (expected_error_duration=0 → div by zero → NaN).
    """

    def test_rejects_zero_error_duration(self):
        with pytest.raises(ValueError, match="finite and positive"):
            HMMDetector(expected_error_duration=0)

    def test_rejects_negative_error_duration(self):
        with pytest.raises(ValueError, match="finite and positive"):
            HMMDetector(expected_error_duration=-5)

    def test_rejects_nan_error_duration(self):
        """NaN passes `<= 0` check but corrupts all HMM calculations."""
        with pytest.raises(ValueError, match="finite and positive"):
            HMMDetector(expected_error_duration=float("nan"))

    def test_rejects_inf_error_duration(self):
        """Inf creates absorbing error state (p_recovery=0 → log(0)=-inf)."""
        with pytest.raises(ValueError, match="finite and positive"):
            HMMDetector(expected_error_duration=float("inf"))

    def test_error_duration_one_does_not_produce_nan(self):
        """expected_error_duration=1 is valid but previously caused -inf in
        transition matrix via log(1 - p_recovery) = log(0). Now p_recovery
        is clamped.
        """
        det = HMMDetector(expected_error_duration=1.0)
        n = 500
        kps = ["nose"]
        error_frames = {"nose": list(range(200, 220))}
        df = _make_bimodal_metric(n, kps, error_frames, error_mean=50.0)

        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        # Must not produce NaN posteriors
        assert not result.posteriors_df["nose"].isna().any(), (
            "NaN posteriors with expected_error_duration=1"
        )


# ---------------------------------------------------------------------------
# compute_ensemble_variance
# ---------------------------------------------------------------------------

# Additional mocks for predictions.py dependencies (torch, lightning, etc.)
# compute_ensemble_variance is pure pandas/numpy but lives in predictions.py
# which has heavy top-level imports.
for _mod in [
    "cv2", "torch", "lightning", "lightning.pytorch",
    "moviepy", "omegaconf", "torchtyping",
    "lightning_pose.callbacks", "lightning_pose.data",
    "lightning_pose.data.dali", "lightning_pose.data.datamodules",
    "lightning_pose.data.utils", "lightning_pose.models",
    "lightning_pose.api", "lightning_pose.api.model",
]:
    sys.modules.setdefault(_mod, mock.MagicMock())

# typeguard must provide a pass-through decorator, not a MagicMock
if "typeguard" not in sys.modules:
    _fake_typeguard = mock.MagicMock()
    _fake_typeguard.typechecked = lambda fn: fn
    sys.modules["typeguard"] = _fake_typeguard
else:
    # If real typeguard is loaded, leave it alone
    pass

_pred_mod = _load_module("lightning_pose.utils.predictions", "predictions.py")
compute_ensemble_variance = _pred_mod.compute_ensemble_variance


def _make_prediction_csv(
    path: Path,
    scorer: str,
    bodyparts: list[str],
    n_frames: int,
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic prediction CSV in DLC 3-level MultiIndex format."""
    rng = np.random.RandomState(seed)
    cols = []
    data = []
    for j, bp in enumerate(bodyparts):
        if x_values is not None:
            x = x_values[:, j] if x_values.ndim > 1 else x_values
        else:
            x = rng.uniform(100, 500, n_frames)
        if y_values is not None:
            y = y_values[:, j] if y_values.ndim > 1 else y_values
        else:
            y = rng.uniform(100, 500, n_frames)
        lh = np.ones(n_frames) * 0.9
        cols.extend([(scorer, bp, "x"), (scorer, bp, "y"), (scorer, bp, "likelihood")])
        data.extend([x, y, lh])
    mi = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(np.column_stack(data), columns=mi)
    df.to_csv(path)
    return df


class TestComputeEnsembleVariance:
    """Tests for compute_ensemble_variance — the core ensemble feature.

    Addresses: AEGIS coverage gap (zero direct tests), VOSS tests section.
    """

    def test_basic_mean_and_variance(self):
        """Correct mean and variance for 3 models with known values."""
        n = 10
        bps = ["nose", "tail"]
        scorer = "heatmap_tracker"  # All models share the same scorer
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for model_idx in range(3):
                p = Path(tmpdir) / f"model_{model_idx}.csv"
                # Model i has x = i+1 for all frames and bodyparts
                x_vals = np.full((n, len(bps)), float(model_idx + 1))
                y_vals = np.full((n, len(bps)), float(model_idx + 1) * 10)
                _make_prediction_csv(
                    p, scorer, bps, n,
                    x_values=x_vals, y_values=y_vals, seed=model_idx,
                )
                paths.append(p)

            mean_preds, variance_df = compute_ensemble_variance(paths)

            # Mean x across [1, 2, 3] = 2.0
            for bp in bps:
                mean_x = mean_preds[("ensemble", bp, "x")].values
                np.testing.assert_allclose(mean_x, 2.0, atol=1e-10)
                # Var([1,2,3], ddof=1) = 1.0
                x_var = variance_df[("ensemble", bp, "x_var")].values
                np.testing.assert_allclose(x_var, 1.0, atol=1e-10)

    def test_rejects_single_csv(self):
        """Must raise ValueError for fewer than 2 CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "single.csv"
            _make_prediction_csv(p, "scorer", ["nose"], 10)
            with pytest.raises(ValueError, match="at least 2"):
                compute_ensemble_variance([p])

    def test_shape_mismatch_raises(self):
        """CSVs with different shapes must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "model_0.csv"
            p2 = Path(tmpdir) / "model_1.csv"
            _make_prediction_csv(p1, "scorer", ["nose"], 10, seed=0)
            _make_prediction_csv(p2, "scorer", ["nose"], 20, seed=1)
            with pytest.raises(ValueError, match="shape mismatch"):
                compute_ensemble_variance([p1, p2])

    def test_nan_propagation(self):
        """NaN in one model's predictions produces NaN variance (not 0)."""
        n = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "model_0.csv"
            p2 = Path(tmpdir) / "model_1.csv"
            p3 = Path(tmpdir) / "model_2.csv"

            # Model 0 and 2 have normal values; model 1 has NaN at frame 5
            for i, p in enumerate([p1, p2, p3]):
                x = np.full((n, 1), float(i + 1))
                y = np.full((n, 1), float(i + 1))
                if i == 1:
                    x[5, 0] = np.nan
                    y[5, 0] = np.nan
                _make_prediction_csv(p, "scorer", ["nose"], n, x_values=x, y_values=y, seed=i)

            mean_preds, variance_df = compute_ensemble_variance([p1, p2, p3])

            # Frame 5: only 2 non-NaN values → nanvar(ddof=1) on [1.0, 3.0] = 2.0
            var_at_5 = variance_df[("ensemble", "nose", "x_var")].iloc[5]
            np.testing.assert_allclose(var_at_5, 2.0, atol=1e-10)

    def test_csv_round_trip(self):
        """Output CSVs can be read back with correct MultiIndex structure."""
        n = 10
        bps = ["nose", "tail"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = Path(tmpdir) / f"model_{i}.csv"
                _make_prediction_csv(p, "scorer", bps, n, seed=i)
                paths.append(p)

            mean_preds, variance_df = compute_ensemble_variance(paths)

            # Save and reload
            mean_path = Path(tmpdir) / "ensemble_mean.csv"
            var_path = Path(tmpdir) / "ensemble_variance.csv"
            mean_preds.to_csv(mean_path)
            variance_df.to_csv(var_path)

            loaded_mean = pd.read_csv(mean_path, header=[0, 1, 2], index_col=0)
            loaded_var = pd.read_csv(var_path, header=[0, 1, 2], index_col=0)

            assert loaded_mean.shape == mean_preds.shape
            assert loaded_var.shape == variance_df.shape
            # Check column structure preserved
            total_var_cols = [c for c in loaded_var.columns if c[2] == "total_var"]
            assert len(total_var_cols) == len(bps)

    def test_mixed_scorer_names(self):
        """Mixed-architecture ensembles have different scorer names (ENTROPY-2).

        heatmap_tracker vs heatmap_mhcrnn_tracker should produce valid variance
        as long as bodyparts and coords match.
        """
        n = 10
        bps = ["nose", "tail"]
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "heatmap_model.csv"
            p2 = Path(tmpdir) / "mhcrnn_model.csv"
            p3 = Path(tmpdir) / "another_heatmap.csv"

            x1 = np.full((n, len(bps)), 1.0)
            x2 = np.full((n, len(bps)), 2.0)
            x3 = np.full((n, len(bps)), 3.0)

            _make_prediction_csv(p1, "heatmap_tracker", bps, n, x_values=x1, seed=0)
            _make_prediction_csv(p2, "heatmap_mhcrnn_tracker", bps, n, x_values=x2, seed=1)
            _make_prediction_csv(p3, "heatmap_tracker", bps, n, x_values=x3, seed=2)

            mean_preds, variance_df = compute_ensemble_variance([p1, p2, p3])

            # Mean x across [1, 2, 3] = 2.0
            for bp in bps:
                mean_x = mean_preds[("ensemble", bp, "x")].values
                np.testing.assert_allclose(mean_x, 2.0, atol=1e-10)

            # Output should use scorer="ensemble" regardless of input scorers
            assert all(c[0] == "ensemble" for c in mean_preds.columns)
            assert all(c[0] == "ensemble" for c in variance_df.columns)


# ---------------------------------------------------------------------------
# Ensemble → QC integration
# ---------------------------------------------------------------------------

class TestEnsembleQCIntegration:
    """Test that compute_ensemble_variance output feeds directly into QC.

    Addresses: AEGIS 6.6 — integration boundary between predictions.py and qc.py.
    """

    def test_ensemble_variance_feeds_qc(self):
        """End-to-end: compute_ensemble_variance output → flag_outlier_frames."""
        n = 500
        bps = ["nose", "tail"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = Path(tmpdir) / f"model_{i}.csv"
                _make_prediction_csv(p, "scorer", bps, n, seed=i)
                paths.append(p)

            _, variance_df = compute_ensemble_variance(paths)

            # Inject a high-variance spike at frame 250
            variance_df.loc[250, ("ensemble", "nose", "x_var")] = 5000.0
            variance_df.loc[250, ("ensemble", "nose", "y_var")] = 5000.0
            total_col = ("ensemble", "nose", "total_var")
            variance_df.loc[250, total_col] = 10000.0

            result = flag_outlier_frames(ensemble_variance_df=variance_df)
            assert "nose" in result.flags_df.columns
            assert "tail" in result.flags_df.columns
            assert result.flags_df.shape[0] == n

    def test_ensemble_variance_csv_feeds_qc_after_round_trip(self):
        """Variance CSV saved and read back still works with QC."""
        n = 300
        bps = ["nose"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = Path(tmpdir) / f"model_{i}.csv"
                _make_prediction_csv(p, "scorer", bps, n, seed=i)
                paths.append(p)

            _, variance_df = compute_ensemble_variance(paths)
            # Save and reload (the actual production interface)
            var_path = Path(tmpdir) / "ensemble_variance.csv"
            variance_df.to_csv(var_path)
            loaded = pd.read_csv(var_path, header=[0, 1, 2], index_col=0)

            result = flag_outlier_frames(ensemble_variance_df=loaded)
            assert result.flags_df.shape == (n, 1)
            assert "nose" in result.flags_df.columns


# ---------------------------------------------------------------------------
# save_qc_report prefix sanitization
# ---------------------------------------------------------------------------

class TestPrefixSanitization:
    """Verify save_qc_report rejects path-traversal prefixes."""

    def test_rejects_path_traversal(self):
        df = _make_temporal_norm(50, ["nose"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="path separator"):
                save_qc_report(result, tmpdir, prefix="../../evil_")

    def test_rejects_backslash_traversal(self):
        df = _make_temporal_norm(50, ["nose"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="path separator"):
                save_qc_report(result, tmpdir, prefix="..\\evil_")

    def test_accepts_normal_prefix(self):
        df = _make_temporal_norm(50, ["nose"], seed=1)
        result = flag_outlier_frames(temporal_norm_df=df)
        with tempfile.TemporaryDirectory() as tmpdir:
            flags_path = save_qc_report(result, tmpdir, prefix="animal_0_")
            assert flags_path.name == "animal_0_qc_flags.csv"


# ---------------------------------------------------------------------------
# Negative metric warning
# ---------------------------------------------------------------------------

class TestNegativeMetricWarning:
    """Negative values in metrics should trigger a warning, not be silently dropped."""

    def test_negative_values_warn(self):
        n = 500
        df = _make_temporal_norm(n, ["nose"], seed=42)
        # Inject negative values
        df.loc[10, "nose"] = -5.0
        df.loc[20, "nose"] = -10.0

        import logging
        with mock.patch.object(
            _qc_mod.logger, "warning"
        ) as mock_warn:
            flag_outlier_frames(temporal_norm_df=df)
            # Should have warned about negative values
            assert mock_warn.called
            warn_msg = mock_warn.call_args[0][0]
            assert "negative" in warn_msg.lower()

    def test_negative_values_in_short_sequence_warn(self):
        """Quantile fallback path should also warn on negative values."""
        n = 50  # below MIN_FRAMES_FOR_HMM
        df = _make_temporal_norm(n, ["nose"], seed=42)
        df.loc[5, "nose"] = -3.0

        import logging
        with mock.patch.object(
            _qc_mod.logger, "warning"
        ) as mock_warn:
            flag_outlier_frames(temporal_norm_df=df)
            assert mock_warn.called


# ---------------------------------------------------------------------------
# Metric frame count mismatch
# ---------------------------------------------------------------------------

class TestMetricFrameCountMismatch:
    """Metrics with different row counts should raise, not silently misalign."""

    def test_mismatched_lengths_raises(self):
        df1 = _make_temporal_norm(200, ["nose"], seed=1)
        df2 = _make_temporal_norm(300, ["nose"], seed=2)
        with pytest.raises(ValueError, match="different row counts"):
            flag_outlier_frames(temporal_norm_df=df1, pca_singleview_df=df2)

    def test_matching_lengths_ok(self):
        df1 = _make_temporal_norm(200, ["nose"], seed=1)
        df2 = _make_temporal_norm(200, ["nose"], seed=2)
        result = flag_outlier_frames(temporal_norm_df=df1, pca_singleview_df=df2)
        assert result.flags_df.shape[0] == 200


# ---------------------------------------------------------------------------
# Boundary-condition tests (AEGIS coverage gaps)
# ---------------------------------------------------------------------------

class TestEnsembleVarianceBoundary:
    """Boundary tests for compute_ensemble_variance."""

    def test_two_models_minimum(self):
        """Two models is the minimum for variance — should work and produce valid output."""
        n = 20
        bps = ["nose", "tail"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(2):
                p = Path(tmpdir) / f"model_{i}.csv"
                x_vals = np.full((n, len(bps)), float(i + 1))
                y_vals = np.full((n, len(bps)), float(i + 1) * 10)
                _make_prediction_csv(p, "scorer", bps, n, x_values=x_vals, y_values=y_vals, seed=i)
                paths.append(p)

            mean_preds, variance_df = compute_ensemble_variance(paths)

            # Mean of [1, 2] = 1.5
            for bp in bps:
                mean_x = mean_preds[("ensemble", bp, "x")].values
                np.testing.assert_allclose(mean_x, 1.5, atol=1e-10)
                # Var([1, 2], ddof=1) = 0.5
                x_var = variance_df[("ensemble", bp, "x_var")].values
                np.testing.assert_allclose(x_var, 0.5, atol=1e-10)

    def test_all_identical_models_zero_variance(self):
        """Identical predictions across models should produce zero variance throughout."""
        n = 50
        bps = ["nose"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            x_vals = np.full((n, len(bps)), 100.0)
            y_vals = np.full((n, len(bps)), 200.0)
            for i in range(3):
                p = Path(tmpdir) / f"model_{i}.csv"
                _make_prediction_csv(p, "scorer", bps, n, x_values=x_vals, y_values=y_vals, seed=i)
                paths.append(p)

            mean_preds, variance_df = compute_ensemble_variance(paths)

            # Variance should be exactly zero
            for bp in bps:
                x_var = variance_df[("ensemble", bp, "x_var")].values
                y_var = variance_df[("ensemble", bp, "y_var")].values
                np.testing.assert_allclose(x_var, 0.0, atol=1e-10)
                np.testing.assert_allclose(y_var, 0.0, atol=1e-10)

            # Zero variance through QC should not crash
            result = flag_outlier_frames(ensemble_variance_df=variance_df)
            assert result.flags_df.shape == (n, len(bps))


class TestHMMBoundary:
    """Boundary tests for HMM detector edge cases."""

    def test_single_frame(self):
        """T=1 sequence should not crash — uses quantile fallback."""
        df = pd.DataFrame({"nose": [5.0]})
        det = HMMDetector()
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        assert result.flags_df.shape == (1, 1)
        # Should not produce NaN
        assert not result.posteriors_df.isna().any().any()

    def test_100_frame_hmm_boundary(self):
        """Exactly 100 frames = MIN_FRAMES_FOR_HMM — should use HMM, not fallback."""
        n = 100  # exactly at boundary
        kps = ["nose"]
        # Create bimodal data so HMM has something to detect
        error_frames = {"nose": list(range(80, 95))}
        df = _make_bimodal_metric(n, kps, error_frames, clean_mean=2.0, error_mean=50.0)

        det = HMMDetector()
        result = flag_outlier_frames(temporal_norm_df=df, detector=det)
        assert result.flags_df.shape == (n, 1)
        # HMM should flag at least some of the error frames
        flagged_in_error = result.flags_df.loc[80:94, "nose"].sum()
        assert flagged_in_error >= 3, f"Only {flagged_in_error}/15 error frames flagged at boundary"
