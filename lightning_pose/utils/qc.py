"""Automated QC flagging of tracking errors.

Reads metric CSVs (temporal norm, PCA reprojection error, ensemble variance)
and flags frames/keypoints where values indicate suspected tracking errors.

The default detector is a two-state Hidden Markov Model with Gaussian mixture
emissions — see :class:`HMMDetector`.

Pure numpy/pandas — no torch or LP-internal imports required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QCResult:
    """Result of QC flagging.

    Attributes:
        flags_df: Boolean DataFrame (n_frames, n_keypoints). True = flagged.
        posteriors_df: Float DataFrame (n_frames, n_keypoints) with anomaly
            scores in [0, 1].  Populated by probabilistic detectors
            (e.g. HMMDetector); ``None`` for binary detectors.
        summary: Dict with ``n_frames``, ``n_flagged_frames``,
            ``pct_flagged_frames``, ``per_keypoint`` counts.
        detector_name: Name of the detector that produced this result.

    """

    flags_df: pd.DataFrame
    posteriors_df: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)
    detector_name: str = ""


def _normalize_ensemble_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Extract total_var columns from MultiIndex ensemble variance DataFrame.

    Converts ``(scorer, bodypart, "total_var")`` MultiIndex columns to flat
    bodypart-named columns suitable for per-keypoint thresholding.

    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    total_var_cols = [c for c in df.columns if c[2] == "total_var"]
    if not total_var_cols:
        return df

    result = pd.DataFrame(index=df.index)
    for col in total_var_cols:
        bodypart = col[1]
        result[bodypart] = df[col].values
    return result


def _normalize_metric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric columns (e.g. 'set') from a flat metric DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].copy()


# ---------------------------------------------------------------------------
# HMM Detector
# ---------------------------------------------------------------------------

def _log_gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log-probability density of a Gaussian (vectorized)."""
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


class HMMDetector:
    """Hidden Markov Model detector with Gaussian mixture emissions.

    Models each keypoint's metric time series as emissions from a two-state
    HMM: state 0 (clean tracking) and state 1 (tracking error).  Emission
    distributions are Gaussians fitted via EM on log-transformed metrics.

    The posterior P(z_t = 1 | all observations) is computed exactly via the
    forward-backward algorithm.  Note: these are posterior probabilities
    under the assumed model, not empirically calibrated probabilities.

    The Gaussian emission family guarantees that each M-step update is
    globally optimal (closed-form sufficient statistics).  However, the
    overall EM trajectory for mixture models can converge to local optima
    of the marginal likelihood; multiple random restarts are used to
    mitigate this.

    When only a single metric is available and the emission model is
    correctly specified, the posterior reduces to a monotone function of
    the likelihood ratio — equivalent to the Neyman-Pearson optimal test.

    Args:
        expected_error_duration: Expected number of consecutive frames in a
            tracking error event.  Controls temporal persistence of the HMM
            states.  Default: 10 frames.
    """

    name = "hmm"

    _EM_ITERATIONS = 30
    _N_RESTARTS = 5
    _MIN_FRAMES_FOR_HMM = 100

    def __init__(self, expected_error_duration: float = 10.0):
        if not np.isfinite(expected_error_duration) or expected_error_duration <= 0:
            raise ValueError(
                f"expected_error_duration must be finite and positive, "
                f"got {expected_error_duration}"
            )
        self.expected_error_duration = expected_error_duration

    def fit_score(
        self,
        metrics: dict[str, pd.DataFrame],
        n_frames: int,
        keypoints: list[str],
    ) -> pd.DataFrame:
        """Score each keypoint's metric time series and return anomaly posteriors.

        Runs an independent HMM (or quantile fallback for short sequences) per
        metric per keypoint, then combines posteriors via noisy-OR.

        Args:
            metrics: Mapping from metric name to DataFrame with keypoint columns.
            n_frames: Total number of video frames.
            keypoints: List of keypoint names to score.

        Returns:
            DataFrame of shape ``(n_frames, len(keypoints))`` with anomaly
            posteriors in [0, 1].

        """
        if n_frames < self._MIN_FRAMES_FOR_HMM:
            logger.info(
                f"Short sequence ({n_frames} frames < {self._MIN_FRAMES_FOR_HMM}), "
                f"using quantile fallback instead of HMM"
            )

        posteriors = pd.DataFrame(
            0.0, index=range(n_frames), columns=keypoints,
        )

        for kp in keypoints:
            # Run independent HMM per metric, then combine posteriors.
            # This correctly handles errors appearing in different metrics
            # at different times (an ID swap might spike temporal_norm at
            # frame 100 but ensemble_variance at frame 300).
            kp_posteriors: list[np.ndarray] = []
            kp_has_metric = False  # track whether any metric covers this kp
            for metric_name, df in metrics.items():
                if kp not in df.columns:
                    continue
                kp_has_metric = True
                vals = df[kp].values.astype(float)
                if np.all(np.isnan(vals)):
                    continue

                if n_frames < self._MIN_FRAMES_FOR_HMM:
                    p = self._quantile_fallback(
                        {metric_name: vals}, n_frames,
                    )
                else:
                    p = self._hmm_inference_single(vals, n_frames)
                kp_posteriors.append(p)

            if kp_posteriors:
                # Noisy-OR: P(any error) = 1 - prod(1 - P_i).
                # Assumes conditional independence of metrics given the true
                # state.  Correlated metrics may inflate the combined posterior.
                combined = np.ones(n_frames)
                for p in kp_posteriors:
                    combined *= (1.0 - p)
                posteriors[kp] = 1.0 - combined
            elif kp_has_metric:
                # All metrics existed but were entirely NaN — no model could
                # track this keypoint.  Default to flagged (1.0), not clean.
                posteriors[kp] = 1.0

        return posteriors

    def _hmm_inference_single(
        self, values: np.ndarray, n_frames: int,
    ) -> np.ndarray:
        """HMM inference for one keypoint on a single metric."""
        valid = np.isfinite(values) & (values >= 0)
        n_negative = int(np.sum(np.isfinite(values) & (values < 0)))
        if n_negative > 0:
            logger.warning(
                f"{n_negative} negative metric values dropped "
                f"(treated as clean); this may indicate a data issue"
            )
        if valid.sum() < 20:
            return np.zeros(n_frames)

        log_vals = np.full(n_frames, np.nan)
        # Add small epsilon to handle exact zeros (e.g. temporal_norm at
        # frame 0, or a perfectly stationary keypoint).
        _eps = 1e-10
        log_vals[valid] = np.log(values[valid] + _eps)

        params = self._fit_gaussian_mixture_em(log_vals[valid])
        if params is None:
            logger.debug("No bimodality detected (BIC prefers 1 component)")
            return np.zeros(n_frames)

        # params are canonical: component 0 = clean, 1 = error (higher mean)
        mu_0, sig_0, mu_1, sig_1, pi_1 = params

        # Emission log-likelihoods
        log_lik = np.zeros((n_frames, 2))
        valid_mask = ~np.isnan(log_vals)
        log_lik[valid_mask, 0] = _log_gaussian_pdf(
            log_vals[valid_mask], mu_0, sig_0,
        )
        log_lik[valid_mask, 1] = _log_gaussian_pdf(
            log_vals[valid_mask], mu_1, sig_1,
        )

        # Transition probabilities derived from mixing weight
        p_recovery = 1.0 / self.expected_error_duration
        # Clamp to prevent log(0) when expected_error_duration=1
        p_recovery = min(p_recovery, 1.0 - 1e-6)
        pi_clipped = float(np.clip(pi_1, 0.001, 0.5))
        p_onset = pi_clipped * p_recovery / (1.0 - pi_clipped)
        p_onset = float(np.clip(p_onset, 1e-6, 0.5))

        return self._forward_backward(log_lik, p_onset, p_recovery)

    def _fit_gaussian_mixture_em(
        self, data: np.ndarray,
    ) -> tuple[float, float, float, float, float] | None:
        """Fit a 2-component Gaussian mixture via EM with BIC model selection.

        Runs multiple random restarts and selects the best fit by
        log-likelihood.  Then compares 1-component vs 2-component models
        using BIC to decide if the data are genuinely bimodal.

        Args:
            data: 1-D array of log-transformed metric values (no NaN).

        Returns:
            ``(mu_0, sigma_0, mu_1, sigma_1, pi_1)`` or ``None`` if the
            data are unimodal (BIC prefers 1 component).

        """
        n = len(data)
        if n < 20:
            return None

        min_sigma = 1e-4

        # --- BIC for 1-component Gaussian (2 params: mu, sigma) ---
        mu_1c = float(np.mean(data))
        sigma_1c = max(min_sigma, float(np.std(data)))
        ll_1c = float(np.sum(_log_gaussian_pdf(data, mu_1c, sigma_1c)))
        bic_1c = -2.0 * ll_1c + 2.0 * np.log(n)

        # --- Multiple EM restarts for 2-component mixture ---
        best_params = None
        best_ll = -np.inf

        data_std = max(min_sigma, float(np.std(data)))
        rng = np.random.RandomState(42)

        for restart in range(self._N_RESTARTS):
            if restart == 0:
                # Deterministic init: median and 99th percentile
                mu_0 = float(np.percentile(data, 50))
                mu_1 = float(np.percentile(data, 99))
                sigma_0 = max(min_sigma, data_std * 0.5)
                sigma_1 = sigma_0
                pi_1 = 0.1
            else:
                # Random perturbation around quantile-based init
                mu_0 = float(np.percentile(data, rng.uniform(30, 60)))
                mu_1 = float(np.percentile(data, rng.uniform(90, 100)))
                sigma_0 = max(min_sigma, data_std * rng.uniform(0.3, 0.8))
                sigma_1 = max(min_sigma, data_std * rng.uniform(0.3, 0.8))
                pi_1 = rng.uniform(0.02, 0.2)

            prev_ll = -np.inf

            for _ in range(self._EM_ITERATIONS):
                # E-step: responsibilities
                ll_0 = np.log(1.0 - pi_1) + _log_gaussian_pdf(data, mu_0, sigma_0)
                ll_1_arr = np.log(pi_1) + _log_gaussian_pdf(data, mu_1, sigma_1)
                ll_max = np.maximum(ll_0, ll_1_arr)
                log_norm = ll_max + np.log(
                    np.exp(ll_0 - ll_max) + np.exp(ll_1_arr - ll_max),
                )

                r_1 = np.exp(ll_1_arr - log_norm)
                r_0 = 1.0 - r_1

                # Convergence check
                total_ll = float(np.sum(log_norm))
                if prev_ll != -np.inf and abs(total_ll - prev_ll) < 1e-6 * abs(prev_ll):
                    break
                prev_ll = total_ll

                # M-step
                n_0 = float(np.sum(r_0))
                n_1 = float(np.sum(r_1))
                if n_0 < 2 or n_1 < 2:
                    break

                mu_0 = float(np.sum(r_0 * data) / n_0)
                mu_1 = float(np.sum(r_1 * data) / n_1)
                sigma_0 = max(
                    min_sigma,
                    float(np.sqrt(np.sum(r_0 * (data - mu_0) ** 2) / n_0)),
                )
                sigma_1 = max(
                    min_sigma,
                    float(np.sqrt(np.sum(r_1 * (data - mu_1) ** 2) / n_1)),
                )
                pi_1 = float(np.clip(n_1 / n, 0.001, 0.999))

            # Canonicalize per-restart: component 0 = clean (lower mean),
            # component 1 = error (higher mean in log-space).
            r_mu_0, r_sig_0, r_mu_1, r_sig_1, r_pi_1 = (
                mu_0, sigma_0, mu_1, sigma_1, pi_1,
            )
            if r_mu_0 > r_mu_1:
                r_mu_0, r_mu_1 = r_mu_1, r_mu_0
                r_sig_0, r_sig_1 = r_sig_1, r_sig_0
                r_pi_1 = 1.0 - r_pi_1

            # Domain knowledge guards (applied after canonicalization):
            # 1. Errors should be a small minority of frames.
            if r_pi_1 > 0.35:
                continue
            # 2. Components must be meaningfully separated — otherwise the
            #    mixture is fitting distributional shape (e.g. skewness),
            #    not genuine bimodality.
            avg_sigma = (r_sig_0 + r_sig_1) / 2.0
            if abs(r_mu_1 - r_mu_0) < 1.5 * avg_sigma:
                continue

            if prev_ll > best_ll and prev_ll != -np.inf:
                best_ll = prev_ll
                best_params = (r_mu_0, r_sig_0, r_mu_1, r_sig_1, r_pi_1)

        if best_params is None:
            return None

        # --- BIC for 2-component mixture (5 params) ---
        bic_2c = -2.0 * best_ll + 5.0 * np.log(n)

        # Prefer 1-component unless 2-component BIC is strictly better
        if bic_2c >= bic_1c:
            return None

        return best_params

    def _forward_backward(
        self,
        log_lik: np.ndarray,
        p_onset: float,
        p_recovery: float,
    ) -> np.ndarray:
        """Forward-backward algorithm for a 2-state HMM.

        Exact marginal inference via the forward-backward algorithm
        (Baum et al., 1970).  Forward and backward messages are summed
        at each node to obtain exact marginal posteriors.

        Returns P(state = 1 | all observations) per frame.
        """
        T = log_lik.shape[0]

        log_A = np.array([
            [np.log(1 - p_onset), np.log(p_onset)],
            [np.log(p_recovery), np.log(1 - p_recovery)],
        ])

        # Stationary initial distribution
        pi_1 = p_onset / (p_onset + p_recovery)
        log_pi = np.array([np.log(1.0 - pi_1), np.log(pi_1)])

        # Forward pass (log_alpha)
        log_alpha = np.zeros((T, 2))
        log_alpha[0] = log_pi + log_lik[0]

        for t in range(1, T):
            log_alpha[t, 0] = log_lik[t, 0] + np.logaddexp(
                log_alpha[t - 1, 0] + log_A[0, 0],
                log_alpha[t - 1, 1] + log_A[1, 0],
            )
            log_alpha[t, 1] = log_lik[t, 1] + np.logaddexp(
                log_alpha[t - 1, 0] + log_A[0, 1],
                log_alpha[t - 1, 1] + log_A[1, 1],
            )

        # Backward pass (log_beta)
        log_beta = np.zeros((T, 2))
        for t in range(T - 2, -1, -1):
            future = log_lik[t + 1] + log_beta[t + 1]
            log_beta[t, 0] = np.logaddexp(
                log_A[0, 0] + future[0],
                log_A[0, 1] + future[1],
            )
            log_beta[t, 1] = np.logaddexp(
                log_A[1, 0] + future[0],
                log_A[1, 1] + future[1],
            )

        # Posterior: P(state=1 | all observations)
        log_gamma = log_alpha + log_beta
        log_evidence = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])

        posteriors = np.exp(log_gamma[:, 1] - log_evidence)
        posteriors = np.clip(posteriors, 0.0, 1.0)
        return posteriors

    def _quantile_fallback(
        self, metric_values: dict[str, np.ndarray], n_frames: int,
    ) -> np.ndarray:
        """Quantile-based scoring for sequences too short for HMM."""
        scores = np.zeros(n_frames)

        for values in metric_values.values():
            valid = np.isfinite(values) & (values >= 0)
            n_negative = int(np.sum(np.isfinite(values) & (values < 0)))
            if n_negative > 0:
                logger.warning(
                    f"{n_negative} negative metric values dropped "
                    f"(treated as clean); this may indicate a data issue"
                )
            if valid.sum() < 5:
                continue
            q90 = float(np.percentile(values[valid], 90))
            q99 = float(np.percentile(values[valid], 99))
            if q99 <= q90 or q99 == 0:
                continue
            metric_scores = np.clip((values - q90) / (q99 - q90), 0.0, 1.0)
            metric_scores[np.isnan(values)] = 0.0
            scores = np.maximum(scores, metric_scores)

        return scores


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def flag_outlier_frames(
    temporal_norm_df: pd.DataFrame | None = None,
    pca_singleview_df: pd.DataFrame | None = None,
    ensemble_variance_df: pd.DataFrame | None = None,
    detector: object | None = None,
    threshold: float = 0.5,
) -> QCResult:
    """Flag frames with suspected tracking errors.

    Normalizes the input metric DataFrames, passes them to the chosen
    detector, and thresholds the resulting anomaly scores into binary flags.

    Args:
        temporal_norm_df: Temporal norm metric DataFrame (optional).
        pca_singleview_df: PCA singleview error DataFrame (optional).
        ensemble_variance_df: Ensemble variance DataFrame with MultiIndex
            columns; only ``total_var`` columns are used (optional).
        detector: A QC detector instance (anything with a ``fit_score``
            method).  Defaults to :class:`HMMDetector`.
        threshold: Score threshold for binary flagging.  Frames with
            anomaly score > threshold are flagged.  Default 0.5.

    Returns:
        :class:`QCResult` with flags, posteriors, and summary statistics.

    Raises:
        ValueError: If no metric DataFrames are provided.

    .. note::

        The default HMM detector is methodologically principled but has not
        been validated against manually-annotated ground-truth tracking errors.
        Before advertising flags as "validated QC," evaluate on at least one
        real annotated dataset with precision-recall analysis across threshold
        values.

    """
    if not (0 < threshold < 1):
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}"
        )

    if detector is None:
        detector = HMMDetector()

    # Normalize inputs
    metrics: dict[str, pd.DataFrame] = {}
    if temporal_norm_df is not None:
        metrics["temporal_norm"] = _normalize_metric_df(temporal_norm_df)
    if pca_singleview_df is not None:
        metrics["pca_singleview"] = _normalize_metric_df(pca_singleview_df)
    if ensemble_variance_df is not None:
        metrics["ensemble_variance"] = _normalize_ensemble_variance(ensemble_variance_df)

    if not metrics:
        raise ValueError("At least one metric DataFrame must be provided")

    # Validate all metrics have the same frame count
    frame_counts = {name: len(df) for name, df in metrics.items()}
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        details = ", ".join(f"{name}={n}" for name, n in frame_counts.items())
        raise ValueError(
            f"Metric DataFrames have different row counts ({details}); "
            f"all metrics must cover the same frames"
        )

    # Determine frame count and keypoint superset
    n_frames = max(len(df) for df in metrics.values())
    all_keypoints: list[str] = []
    seen: set[str] = set()
    for df in metrics.values():
        for col in df.columns:
            if col not in seen:
                all_keypoints.append(col)
                seen.add(col)

    # Run detector
    scores_df = detector.fit_score(metrics, n_frames, all_keypoints)

    # Validate detector output shape
    expected_shape = (n_frames, len(all_keypoints))
    if scores_df.shape != expected_shape:
        raise ValueError(
            f"Detector returned shape {scores_df.shape}, "
            f"expected {expected_shape}"
        )

    # Threshold into binary flags.
    # NaN posteriors (from numeric bugs) must be flagged, not treated as clean:
    # NaN > threshold evaluates to False in numpy, which would silently mark
    # the most suspect frames as "definitely clean."
    flags_df = scores_df > threshold
    nan_mask = scores_df.isna()
    if nan_mask.any().any():
        n_nan = int(nan_mask.sum().sum())
        logger.warning(
            f"Detector produced {n_nan} NaN posterior(s) — flagging as suspect"
        )
        flags_df = flags_df | nan_mask

    # Build summary
    n_flagged_frames = int(flags_df.any(axis=1).sum())
    per_keypoint = {col: int(flags_df[col].sum()) for col in all_keypoints}

    summary = {
        "n_frames": n_frames,
        "n_flagged_frames": n_flagged_frames,
        "pct_flagged_frames": (
            round(100.0 * n_flagged_frames / n_frames, 2)
            if n_frames > 0
            else 0.0
        ),
        "per_keypoint": per_keypoint,
        "threshold": threshold,
    }

    detector_name = getattr(detector, "name", type(detector).__name__)
    logger.info(
        f"QC [{detector_name}]: {n_flagged_frames}/{n_frames} frames flagged "
        f"({summary['pct_flagged_frames']}%)"
    )

    return QCResult(
        flags_df=flags_df,
        posteriors_df=scores_df,
        summary=summary,
        detector_name=detector_name,
    )


def save_qc_report(
    qc_result: QCResult,
    output_dir: str | Path,
    prefix: str = "",
) -> Path:
    """Save QC flags, posteriors, and summary to CSV files.

    Writes:
        - ``{prefix}qc_flags.csv``: 0/1 integer DataFrame
        - ``{prefix}qc_posteriors.csv``: float scores in [0, 1] (if available)
        - ``{prefix}qc_summary.csv``: per-keypoint flagged counts

    Args:
        qc_result: :class:`QCResult` from :func:`flag_outlier_frames`.
        output_dir: Directory to write files to.
        prefix: Optional filename prefix (e.g. ``"animal_0_"``).

    Returns:
        Path to the flags CSV file.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize prefix to prevent path traversal
    if prefix and ("/" in prefix or "\\" in prefix or ".." in prefix):
        raise ValueError(f"prefix contains path separator or traversal: {prefix!r}")

    # Flags (0/1)
    flags_file = output_dir / f"{prefix}qc_flags.csv"
    qc_result.flags_df.astype(int).to_csv(flags_file)

    # Posteriors (float scores)
    if qc_result.posteriors_df is not None:
        posteriors_file = output_dir / f"{prefix}qc_posteriors.csv"
        qc_result.posteriors_df.round(4).to_csv(posteriors_file)

    # Summary
    rows = []
    for kp, count in qc_result.summary["per_keypoint"].items():
        n = qc_result.summary["n_frames"]
        pct = round(100.0 * count / n, 2) if n > 0 else 0.0
        rows.append({
            "keypoint": kp,
            "n_flagged": count,
            "pct_flagged": pct,
            "detector": qc_result.detector_name,
            "threshold": qc_result.summary.get("threshold", ""),
        })

    summary_df = pd.DataFrame(rows)
    summary_file = output_dir / f"{prefix}qc_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"QC report saved to {output_dir}")
    return flags_file


def format_qc_summary(qc_result: QCResult) -> str:
    """Format QC results as a human-readable multi-line string.

    Example output::

        QC [hmm]: 47/1200 frames flagged (3.9%)
          nose:      12 flagged (1.0%)
          left_ear:   8 flagged (0.7%)
          right_ear: 35 flagged (2.9%)

    """
    s = qc_result.summary
    n_frames = s["n_frames"]
    det = f" [{qc_result.detector_name}]" if qc_result.detector_name else ""
    threshold = s.get("threshold", "")
    threshold_str = f", threshold={threshold}" if threshold != "" else ""
    lines = [
        f"QC{det}: {s['n_flagged_frames']}/{n_frames} frames flagged "
        f"({s['pct_flagged_frames']}%{threshold_str})",
    ]

    if s["n_flagged_frames"] == 0:
        lines.append("  No suspected tracking errors detected.")
        return "\n".join(lines)

    per_kp = s["per_keypoint"]
    if per_kp:
        max_len = max(len(kp) for kp in per_kp)
        high_flag_kps = []
        for kp, count in per_kp.items():
            pct = round(100.0 * count / n_frames, 1) if n_frames > 0 else 0.0
            lines.append(f"  {kp:{max_len}s}: {count:>4d} flagged ({pct}%)")
            if n_frames > 0 and count / n_frames > 0.3:
                high_flag_kps.append(kp)
        if high_flag_kps:
            lines.append(
                f"  WARNING: {', '.join(high_flag_kps)} flagged >30% of frames — "
                f"consider relabeling or checking keypoint visibility"
            )

    return "\n".join(lines)
