##########################################
Ensemble predictions and quality control
##########################################

When you train multiple models on the same dataset (e.g. with different random seeds),
you can run all of them on the same video and average their predictions.
Where models disagree, the keypoint is less reliable.
Lightning Pose can flag these unreliable keypoints automatically.

Ensemble inference
==================

To run ensemble inference, pass multiple model directories to ``litpose predict``:

.. code-block:: bash

    litpose predict model_0/ model_1/ model_2/ video.mp4 --output_dir results/

This will:

1. Run each model on the video sequentially (models are loaded and unloaded
   one at a time to avoid GPU memory issues).
2. Compute the mean prediction and per-keypoint variance across models.
3. Run automatic quality control (QC) to flag suspected tracking errors.

For multi-animal workflows using SAM-derived bounding boxes, pass multiple
model directories to ``litpose sam-detect infer``:

.. code-block:: bash

    litpose sam-detect infer \
        --bbox_dir path/to/bboxes/ \
        --video path/to/video.mp4 \
        --model_dir model_0/ model_1/ model_2/ \
        --output_dir results/

Output files
============

Ensemble mode produces the following files per video (or per animal in SAM mode):

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - File
     - Contents
     - When to use
   * - ``ensemble_mean.csv``
     - Averaged x/y predictions across all models (DLC format).
       Likelihood values are also averaged; they reflect mean model confidence,
       not calibrated probabilities.
     - **Primary prediction file.** Feed this to downstream tools
       (keypoint-MoSeq, SimBA, VAME, etc.).
   * - ``ensemble_variance.csv``
     - Per-keypoint disagreement between models (x_var, y_var, total_var),
       in units of pixels\ :sup:`2`. Higher values = less reliable.
     - Diagnostic. Inspect in pandas to identify unreliable keypoints.
   * - ``qc_flags.csv``
     - Binary flags (0 = OK, 1 = suspected error) per frame per keypoint.
     - Use to filter unreliable keypoints before downstream analysis.
   * - ``qc_posteriors.csv``
     - Continuous anomaly scores in [0, 1] per frame per keypoint.
     - Advanced: apply your own threshold instead of the default 0.5.
   * - ``qc_summary.csv``
     - Per-keypoint flagged frame counts, plus detector and threshold used.
     - Quick overview of which bodyparts are problematic.
   * - ``provenance.json``
     - Model directories, software versions, timestamp, and per-model
       success/failure status.
     - Audit trail. Verify which models and code version produced the outputs.

.. note::

    ``ensemble_mean.csv`` contains **unfiltered** predictions. QC flags are in
    separate files — you must apply them yourself (see below).

Applying QC flags
=================

The QC module flags suspected tracking errors but does not modify the
prediction files. To set flagged keypoints to NaN before feeding to
downstream tools:

.. code-block:: python

    import pandas as pd

    preds = pd.read_csv("ensemble_mean.csv", header=[0, 1, 2], index_col=0)
    flags = pd.read_csv("qc_flags.csv", index_col=0)

    for bodypart in flags.columns:
        flagged = flags[bodypart] == 1
        preds.loc[flagged, (slice(None), bodypart, "x")] = float("nan")
        preds.loc[flagged, (slice(None), bodypart, "y")] = float("nan")
        preds.loc[flagged, (slice(None), bodypart, "likelihood")] = float("nan")

    preds.to_csv("ensemble_mean_filtered.csv")

.. warning::

    Setting flagged keypoints to NaN produces **gaps** in the time series.
    Some downstream tools (e.g. keypoint-MoSeq, DeepLabCut) will automatically
    interpolate these gaps, producing plausible-looking but **synthetic**
    trajectories at flagged frames. If your analysis is sensitive to exact
    coordinates (e.g. measuring velocities or distances), either exclude
    flagged frames entirely or verify that your interpolation method is
    appropriate. Do not treat interpolated values as measured data.

Interpreting QC output
======================

When QC runs, it prints a summary to the console:

.. code-block:: text

    QC [hmm]: 47/1200 frames flagged (3.9%)
      nose:      12 flagged (1.0%)
      left_ear:   8 flagged (0.7%)
      right_ear: 35 flagged (2.9%)

A flag rate of **1–10%** per keypoint is typical for videos with occasional
tracking errors. Use the per-keypoint breakdown to identify which bodyparts
are problematic.

**If a keypoint is flagged >30% of frames**, the console will print a warning.
This usually means:

* The keypoint is frequently occluded and should be relabeled.
* The keypoint is not consistently visible in your videos.
* Your ensemble needs retraining with more diverse training data.

**If zero frames are flagged**, the console will print
"No suspected tracking errors detected." This means the HMM found no
evidence of bimodal tracking behavior — your predictions are likely clean.

Disabling QC
============

QC runs automatically in ensemble mode. To skip it (e.g. to avoid extra
output files in a batch pipeline), pass ``--skip_qc``:

.. code-block:: bash

    litpose predict model_0/ model_1/ model_2/ video.mp4 \
        --output_dir results/ --skip_qc

    litpose sam-detect infer \
        --model_dir model_0/ model_1/ model_2/ \
        --video video.mp4 --output_dir results/ --skip_qc

When to use ensemble inference
==============================

**Use ensemble inference when:**

* You have trained 2 or more models (e.g. same config, different random seeds)
  and want to identify unreliable keypoints automatically.
* You want a single averaged prediction file that is more robust than any
  individual model.

**Do not use when:**

* You have only 1 model — ensemble variance requires at least 2 models.
* You need exact reproducibility with prior single-model results — ensemble
  mode produces different output files with ``scorer="ensemble"`` in the header.

.. note::

    Ensemble output CSVs use ``scorer="ensemble"`` in the first level of the
    column MultiIndex. Downstream tools that hard-code the scorer name (e.g.
    ``preds[("heatmap_tracker", "nose", "x")]``) will raise ``KeyError``.
    To fix, either use the bodypart level directly::

        # Works with any scorer name
        x_cols = [c for c in preds.columns if c[1] == "nose" and c[2] == "x"]

    or rename the scorer after loading::

        preds.columns = preds.columns.set_levels(["your_scorer"], level=0)

Known limitations
=================

* **Multi-view models are not supported** in ensemble mode. Each model must be
  a single-view model.
* **QC parameters are not configurable via CLI or config.** The threshold (0.5)
  and temporal persistence (10 frames) are hardcoded defaults. For custom
  thresholds, use the continuous scores in ``qc_posteriors.csv`` and apply
  your own cutoff in Python.
* **QC works best on videos with >100 frames.** Shorter sequences (fewer than
  100 frames) automatically use a simpler quantile-based fallback instead of
  the HMM. The fallback flags frames where the metric exceeds the 90th
  percentile, scaled to [0, 1] by the 90th–99th percentile range.
* **QC may under-flag isolated single-frame errors** and over-flag unusual but
  correct poses. The HMM is tuned for detecting sustained tracking failures
  (e.g. identity swaps) rather than brief glitches.
* **Sparse model coverage is not flagged.** If only 1 of N models produces a
  prediction for a keypoint on a given frame (the others return NaN), the
  variance is undefined and the HMM treats the frame as missing data —
  inferring its state from temporal context rather than flagging it. In
  practice, sparse coverage often indicates an occluded or difficult keypoint.
  Check ``ensemble_variance.csv`` for NaN values if you suspect this affects
  your data.
* **Behavioral state transitions may cause over-flagging.** Videos where the
  animal transitions between stationary and moving states produce natural
  increases in temporal norm during movement. The HMM may misidentify these
  as tracking errors because it cannot distinguish "animal started moving"
  from "tracker jumped." If you see systematic over-flagging during movement
  bouts, consider using ``--skip_qc`` and applying a custom threshold on the
  continuous scores in ``qc_posteriors.csv``.
* **The HMM is fitted independently per video.** Parameters (emission means,
  transition probabilities) are estimated from each video's metric distribution
  via EM. This means QC thresholds adapt to each video's characteristics, but
  also that results are not directly comparable across videos with very
  different tracking quality.
* **Ensemble QC is CLI-only.** The Lightning Pose GUI does not currently
  support ensemble inference or QC visualization. Use the CLI commands
  documented above and inspect QC output files in Python or a spreadsheet.
