# -*- coding: utf-8 -*-
"""
Singapore two-night ABPM-anchored PPG phenotype pipeline
========================================================

Purpose
-------
Using the output of ``sleep_bp_ppg_pipeline_FINAL_FIXED.py``:

1. Build a quality-controlled PPG profile for each subject-night.
2. Derive interpretable PPG morphology/timing variables:
   - reflection index (RI)
   - approximate reflected-wave delay (Delta-T)
   - approximate stiffness index (SI; PPG + height only)
   - P2/P1 and PPG augmentation index (AI) as explicit transforms of RI
   - pulse-rate, rise-time, width, amplitude variability, and stage modulation
3. Quantify feature-level repeatability between the ABPM and cuff-free nights.
4. Test whether each cuff-free-night PPG profile resembles its own
   ABPM-night profile more than other participants' profiles.
5. Construct cross-fitted ABPM-anchored phenotype axes:
   - nocturnal MAP level
   - MAP average real variability (ARV)
   - mean pulse pressure
6. Apply models trained only on other participants' ABPM nights to both nights.
7. Assess whether the same participant receives a similar phenotype score on
   the ABPM and non-ABPM nights.
8. Perform optional exploratory cluster transfer using fixed ABPM-night centroids.

Important interpretation
------------------------
- The non-ABPM night has no BP reference. The analysis tests persistence of an
  ABPM-anchored PPG phenotype, not repeatability of BP itself.
- ``ppg_ai_pct`` is defined as (P2 - P1) / P1 * 100. Because the source
  extractor stores RI = P2 / P1, AI is a deterministic transform of RI and is
  not entered together with RI in prediction models.
- ``ppg_delta_t_ms_approx`` is reconstructed from the window-median normalized
  Delta-T and median pulse period. It is an approximation.
- ``ppg_si_m_s_approx`` requires height and is therefore PPG + static metadata,
  not strict PPG-only.
- APG a/b/c/d/e ratios are not reconstructed from summary CSV files. They
  require beat-level raw-waveform re-extraction and manual QC of landmarks.

Expected input
--------------
BASE_OUTPUT_DIR/
    01_night_processing_inventory.csv
    10_extracted/
        all_stable_stage_windows.csv
        all_bp_linked_pre_cuff_windows.csv

These files are produced by the previously validated Singapore raw pipeline.

No argparse is used. Edit only the USER SETTINGS section and run in Colab.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# USER SETTINGS: edit only this section
# =============================================================================

BASE_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/SG/_SLEEP_BP_PPG_ANALYSIS"
)

PHENOTYPE_OUTPUT_DIR = BASE_OUTPUT_DIR / "60_ABPM_ANCHORED_PPG_PHENOTYPE"

# Optional demographic table. SI is calculated only when a valid height column
# is found. Set to None to run strict PPG-only analysis without SI.
DEMOGRAPHIC_FILE: Optional[Path] = Path(
    "/content/drive/MyDrive/Colab Notebooks/SG/DemographicInfo.csv"
)

RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 2000

# Primary quality control
STAGE_PURITY_MIN = 0.80
STAGE_COVERAGE_MIN = 0.80
PPG_SQI_MIN = 0.75
PPG_VALID_PULSE_FRACTION_MIN = 0.70
MIN_PULSES_PER_WINDOW = 20
PULSE_RATE_MIN = 35.0
PULSE_RATE_MAX = 140.0
EXCLUDE_CUFF_NEARBY_WINDOWS = True

# Subject-level requirements
MIN_WINDOWS_PER_STAGE = 1
MIN_NOCTURNAL_BP_READINGS = 6
MIN_FEATURE_COVERAGE = 0.55
MIN_REPEATABILITY_PAIRS = 15
MIN_TRAIN_ICC_A = 0.20
MAX_MODEL_FEATURES = 20
MIN_MODEL_FEATURES = 5
CORRELATION_PRUNE_THRESHOLD = 0.90

# Cluster analysis is exploratory only.
RUN_EXPLORATORY_CLUSTER_TRANSFER = True
CLUSTER_K_VALUES = (2, 3, 4)
MIN_CLUSTER_SIZE = 8
MIN_SILHOUETTE = 0.20


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PhenotypeConfig:
    base_output_dir: Path
    output_dir: Path
    demographic_file: Optional[Path] = None
    random_state: int = RANDOM_STATE
    n_outer_folds: int = N_OUTER_FOLDS
    n_inner_folds: int = N_INNER_FOLDS
    n_bootstrap: int = N_BOOTSTRAP
    n_permutations: int = N_PERMUTATIONS
    stage_purity_min: float = STAGE_PURITY_MIN
    stage_coverage_min: float = STAGE_COVERAGE_MIN
    ppg_sqi_min: float = PPG_SQI_MIN
    valid_pulse_fraction_min: float = PPG_VALID_PULSE_FRACTION_MIN
    min_pulses: int = MIN_PULSES_PER_WINDOW
    pulse_rate_min: float = PULSE_RATE_MIN
    pulse_rate_max: float = PULSE_RATE_MAX
    exclude_cuff_nearby: bool = EXCLUDE_CUFF_NEARBY_WINDOWS
    min_windows_per_stage: int = MIN_WINDOWS_PER_STAGE
    min_bp_readings: int = MIN_NOCTURNAL_BP_READINGS
    min_feature_coverage: float = MIN_FEATURE_COVERAGE
    min_repeatability_pairs: int = MIN_REPEATABILITY_PAIRS
    min_train_icc_a: float = MIN_TRAIN_ICC_A
    max_model_features: int = MAX_MODEL_FEATURES
    min_model_features: int = MIN_MODEL_FEATURES
    corr_prune_threshold: float = CORRELATION_PRUNE_THRESHOLD
    run_cluster_transfer: bool = RUN_EXPLORATORY_CLUSTER_TRANSFER
    cluster_k_values: Tuple[int, ...] = CLUSTER_K_VALUES
    min_cluster_size: int = MIN_CLUSTER_SIZE
    min_silhouette: float = MIN_SILHOUETTE

    def __post_init__(self) -> None:
        self.base_output_dir = Path(self.base_output_dir)
        self.output_dir = Path(self.output_dir)
        if self.demographic_file is not None:
            self.demographic_file = Path(self.demographic_file)


# Features already available in the validated Singapore extraction output.
SOURCE_PPG_FEATURES = [
    "ppg_amp_median",
    "ppg_amp_cv_pct",
    "ppg_rise_median_ms",
    "ppg_width50_median_ms",
    "ppg_reflection_index_median",
    "ppg_delta_t_norm_median",
    "ppg_pulse_rate_bpm",
]

DERIVED_PPG_FEATURES = [
    "ppg_log_amp",
    "ppg_amp_log_n2_relative",
    "ppg_delta_t_ms_approx",
    "ppg_p2_p1_pct",
    "ppg_ai_pct",
    "ppg_si_m_s_approx",
]

ANALYSIS_STAGES = ("N2", "N3", "REM")

# Strict PPG-only variables. AI is omitted because it is a linear transform of RI.
STRICT_MODEL_BASE_FEATURES = [
    "ppg_amp_log_n2_relative",
    "ppg_amp_cv_pct",
    "ppg_rise_median_ms",
    "ppg_width50_median_ms",
    "ppg_reflection_index_median",
    "ppg_delta_t_norm_median",
    "ppg_delta_t_ms_approx",
    "ppg_pulse_rate_bpm",
]

# PPG + one-time height metadata.
HEIGHT_AUGMENTED_BASE_FEATURES = STRICT_MODEL_BASE_FEATURES + [
    "ppg_si_m_s_approx",
]

BP_TARGETS = {
    "map_level": "abpm_mean_map",
    "map_variability": "abpm_map_arv",
    "pulse_pressure": "abpm_mean_pp",
}


# =============================================================================
# Generic helpers
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
    }
    return (
        series.astype(str).str.strip().str.lower().map(mapping).fillna(False).astype(bool)
    )


def normalize_subject_id(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE"}:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if text.startswith("NUS") and digits:
        return f"NUS{int(digits):04d}"
    if digits and len(digits) >= 3:
        return f"NUS{int(digits):04d}"
    return text


def robust_z(values: pd.Series, center: Optional[float] = None,
             scale: Optional[float] = None) -> Tuple[pd.Series, float, float]:
    x = pd.to_numeric(values, errors="coerce")
    if center is None:
        center = float(np.nanmedian(x))
    if scale is None:
        q25, q75 = np.nanpercentile(x, [25, 75])
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(x, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
    return (x - center) / scale, center, scale


def safe_corr(x: Sequence[float], y: Sequence[float], method: str = "pearson") -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(pair["x"], pair["y"]).statistic)
    return float(stats.pearsonr(pair["x"], pair["y"]).statistic)


def concordance_correlation_coefficient(x: Sequence[float],
                                        y: Sequence[float]) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return np.nan
    xv = pair["x"].to_numpy(dtype=float)
    yv = pair["y"].to_numpy(dtype=float)
    vx = np.var(xv, ddof=1)
    vy = np.var(yv, ddof=1)
    cov = np.cov(xv, yv, ddof=1)[0, 1]
    denominator = vx + vy + (np.mean(xv) - np.mean(yv)) ** 2
    return float(2.0 * cov / denominator) if denominator > 0 else np.nan


def average_real_variability(values: Sequence[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))))


def iqr(values: Sequence[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan, dtype=float)
    valid = denominator > 1e-12
    out[valid] = numerator[valid] / denominator[valid]
    return out


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = a / np.where(a_norm > 1e-12, a_norm, 1.0)
    b_safe = b / np.where(b_norm > 1e-12, b_norm, 1.0)
    return a_safe @ b_safe.T


def tertile_labels(values: pd.Series, reference: Optional[pd.Series] = None) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    ref = x if reference is None else pd.to_numeric(reference, errors="coerce")
    ref = ref.dropna()
    if len(ref) < 6 or ref.nunique() < 3:
        return pd.Series(np.nan, index=x.index, dtype="float")
    q1, q2 = np.nanpercentile(ref, [33.3333, 66.6667])
    labels = pd.Series(np.nan, index=x.index, dtype="float")
    labels.loc[x <= q1] = 0
    labels.loc[(x > q1) & (x <= q2)] = 1
    labels.loc[x > q2] = 2
    return labels


# =============================================================================
# Data loading and demographic merge
# =============================================================================

def resolve_input_path(base_dir: Path, filename: str) -> Path:
    candidates = [
        base_dir / "10_extracted" / filename,
        base_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def load_inputs(cfg: PhenotypeConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stable_path = resolve_input_path(
        cfg.base_output_dir, "all_stable_stage_windows.csv"
    )
    bp_path = resolve_input_path(
        cfg.base_output_dir, "all_bp_linked_pre_cuff_windows.csv"
    )
    inventory_path = resolve_input_path(
        cfg.base_output_dir, "01_night_processing_inventory.csv"
    )

    stable = pd.read_csv(stable_path, low_memory=False)
    bp = pd.read_csv(bp_path, low_memory=False)
    inventory = pd.read_csv(inventory_path, low_memory=False)

    for frame in (stable, bp, inventory):
        if "subject_id" in frame.columns:
            frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
        if "ABPM_exposure" in frame.columns:
            frame["ABPM_exposure"] = normalize_bool(frame["ABPM_exposure"])

    return stable, bp, inventory


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {
        "".join(ch for ch in str(col).lower() if ch.isalnum()): col
        for col in frame.columns
    }
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def load_demographics(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["subject_id", "height_cm"])
    demo = pd.read_csv(path, low_memory=False)

    id_col = find_column(
        demo,
        ["subject_id", "subject", "participant_id", "participant", "record_id", "id"],
    )
    height_col = find_column(
        demo,
        ["height_cm", "height", "body_height_cm", "stature_cm", "ht_cm"],
    )

    if id_col is None:
        warnings.warn(
            f"Demographic file found but no subject ID column was recognized: {path}"
        )
        return pd.DataFrame(columns=["subject_id", "height_cm"])

    out = pd.DataFrame()
    out["subject_id"] = demo[id_col].map(normalize_subject_id)

    if height_col is not None:
        height = pd.to_numeric(demo[height_col], errors="coerce")
        # Convert likely metre values to cm.
        height = height.where(~height.between(1.2, 2.2), height * 100.0)
        height = height.where(height.between(120, 220))
        out["height_cm"] = height
    else:
        out["height_cm"] = np.nan

    return out.drop_duplicates("subject_id")


# =============================================================================
# PPG QC and derived features
# =============================================================================

def apply_ppg_qc(stable: pd.DataFrame, cfg: PhenotypeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = stable.copy()

    required = [
        "subject_id", "night", "ABPM_exposure", "dominant_stage",
        "stage_purity", "stage_coverage", "ppg_sqi",
        "ppg_valid_pulse_fraction", "ppg_n_pulses",
        "ppg_pulse_rate_bpm",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Stable-window table is missing required columns: {missing}")

    numeric_cols = set(SOURCE_PPG_FEATURES + [
        "stage_purity", "stage_coverage", "ppg_sqi",
        "ppg_valid_pulse_fraction", "ppg_n_pulses",
        "motion_robust_z",
    ])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    qc = pd.Series(True, index=df.index)
    qc &= df["dominant_stage"].isin(ANALYSIS_STAGES)
    qc &= df["stage_purity"].ge(cfg.stage_purity_min)
    qc &= df["stage_coverage"].ge(cfg.stage_coverage_min)
    qc &= df["ppg_sqi"].ge(cfg.ppg_sqi_min)
    qc &= df["ppg_valid_pulse_fraction"].ge(cfg.valid_pulse_fraction_min)
    qc &= df["ppg_n_pulses"].ge(cfg.min_pulses)
    qc &= df["ppg_pulse_rate_bpm"].between(
        cfg.pulse_rate_min, cfg.pulse_rate_max
    )

    if "motion_ok" in df.columns:
        qc &= normalize_bool(df["motion_ok"])
    elif "motion_robust_z" in df.columns:
        qc &= df["motion_robust_z"].abs().le(3.5) | df["motion_robust_z"].isna()

    if cfg.exclude_cuff_nearby and "cuff_nearby" in df.columns:
        qc &= ~normalize_bool(df["cuff_nearby"])

    df["phenotype_qc_pass"] = qc

    inventory = (
        df.groupby(["subject_id", "night", "ABPM_exposure"], observed=True)
        .agg(
            n_all_windows=("phenotype_qc_pass", "size"),
            n_qc_windows=("phenotype_qc_pass", "sum"),
        )
        .reset_index()
    )
    inventory["qc_window_fraction"] = (
        inventory["n_qc_windows"] / inventory["n_all_windows"].clip(lower=1)
    )

    return df.loc[qc].copy(), inventory


def add_derived_ppg_features(df: pd.DataFrame,
                             demographics: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(demographics, on="subject_id", how="left")

    amp = pd.to_numeric(out.get("ppg_amp_median"), errors="coerce")
    out["ppg_log_amp"] = np.log(amp.where(amp > 0))

    rate = pd.to_numeric(out.get("ppg_pulse_rate_bpm"), errors="coerce")
    period_ms = 60000.0 / rate.where(rate > 0)
    dtnorm = pd.to_numeric(out.get("ppg_delta_t_norm_median"), errors="coerce")
    out["ppg_delta_t_ms_approx"] = dtnorm * period_ms

    ri = pd.to_numeric(out.get("ppg_reflection_index_median"), errors="coerce")
    out["ppg_p2_p1_pct"] = ri * 100.0
    out["ppg_ai_pct"] = (ri - 1.0) * 100.0

    height_m = pd.to_numeric(out.get("height_cm"), errors="coerce") / 100.0
    delta_t_sec = out["ppg_delta_t_ms_approx"] / 1000.0
    out["ppg_si_m_s_approx"] = height_m / delta_t_sec.where(delta_t_sec > 0)

    # Conservative plausibility limits. Out-of-range values become missing.
    out["ppg_delta_t_ms_approx"] = out["ppg_delta_t_ms_approx"].where(
        out["ppg_delta_t_ms_approx"].between(60, 700)
    )
    out["ppg_si_m_s_approx"] = out["ppg_si_m_s_approx"].where(
        out["ppg_si_m_s_approx"].between(1.0, 30.0)
    )
    out["ppg_reflection_index_median"] = ri.where(ri.between(0.02, 1.50))
    out["ppg_p2_p1_pct"] = out["ppg_p2_p1_pct"].where(
        out["ppg_p2_p1_pct"].between(2, 150)
    )
    out["ppg_ai_pct"] = out["ppg_ai_pct"].where(
        out["ppg_ai_pct"].between(-98, 50)
    )

    # Night-specific N2 amplitude reference. This removes arbitrary sensor gain
    # and emphasizes stage modulation rather than absolute amplitude.
    n2_reference = (
        out.loc[out["dominant_stage"].eq("N2")]
        .groupby(["subject_id", "night"], observed=True)["ppg_log_amp"]
        .median()
        .rename("ppg_log_amp_n2_reference")
        .reset_index()
    )
    out = out.merge(n2_reference, on=["subject_id", "night"], how="left")
    out["ppg_amp_log_n2_relative"] = (
        out["ppg_log_amp"] - out["ppg_log_amp_n2_reference"]
    )
    return out


# =============================================================================
# Subject-night PPG profile
# =============================================================================

def summarize_subject_night_ppg(
    windows: pd.DataFrame,
    cfg: PhenotypeConfig,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    base_features = [
        feature for feature in
        list(dict.fromkeys(SOURCE_PPG_FEATURES + DERIVED_PPG_FEATURES))
        if feature in windows.columns
    ]

    group_cols = ["subject_id", "night", "ABPM_exposure", "dominant_stage"]

    stage_counts = (
        windows.groupby(group_cols, observed=True)
        .size()
        .rename("n_windows")
        .reset_index()
    )

    stage_median = (
        windows.groupby(group_cols, observed=True)[base_features]
        .median()
        .reset_index()
    )
    stage_median = stage_median.merge(stage_counts, on=group_cols, how="left")

    # Require a minimum number of windows for each reported stage estimate.
    feature_cols = [c for c in stage_median.columns if c in base_features]
    insufficient = stage_median["n_windows"].lt(cfg.min_windows_per_stage)
    stage_median.loc[insufficient, feature_cols] = np.nan

    index_cols = ["subject_id", "night", "ABPM_exposure"]
    wide_parts: List[pd.DataFrame] = []

    for feature in base_features:
        pivot = stage_median.pivot_table(
            index=index_cols,
            columns="dominant_stage",
            values=feature,
            aggfunc="first",
        )
        pivot.columns = [f"{feature}__{stage}" for stage in pivot.columns]
        wide_parts.append(pivot)

    count_pivot = stage_counts.pivot_table(
        index=index_cols,
        columns="dominant_stage",
        values="n_windows",
        aggfunc="sum",
        fill_value=0,
    )
    count_pivot.columns = [f"n_windows__{stage}" for stage in count_pivot.columns]
    wide_parts.append(count_pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    # Stage contrasts relative to N2.
    contrast_columns: List[str] = []
    for feature in base_features:
        n2 = f"{feature}__N2"
        for stage in ("N3", "REM"):
            stage_col = f"{feature}__{stage}"
            contrast_col = f"{feature}__{stage}_minus_N2"
            if n2 in wide.columns and stage_col in wide.columns:
                wide[contrast_col] = wide[stage_col] - wide[n2]
                contrast_columns.append(contrast_col)

    # Whole-night robust level and dispersion.
    night_group = windows.groupby(
        ["subject_id", "night", "ABPM_exposure"], observed=True
    )
    night_median = night_group[base_features].median()
    night_median.columns = [f"{c}__night_median" for c in night_median.columns]
    night_iqr = night_group[base_features].agg(iqr)
    night_iqr.columns = [f"{c}__night_iqr" for c in night_iqr.columns]
    night_stats = pd.concat([night_median, night_iqr], axis=1).reset_index()
    wide = wide.merge(
        night_stats,
        on=["subject_id", "night", "ABPM_exposure"],
        how="left",
    )

    # Candidate model variables:
    # - N2 baseline for timing/morphology/rate
    # - N3-N2 and REM-N2 modulation
    # - whole-night IQR
    candidate_columns: List[str] = []
    strict_bases = [
        feature for feature in STRICT_MODEL_BASE_FEATURES
        if feature in base_features
    ]
    for feature in strict_bases:
        for suffix in (
            "__N2",
            "__N3_minus_N2",
            "__REM_minus_N2",
            "__night_iqr",
        ):
            col = f"{feature}{suffix}"
            if col in wide.columns:
                # N2-relative amplitude has a structural zero at N2.
                if feature == "ppg_amp_log_n2_relative" and suffix == "__N2":
                    continue
                candidate_columns.append(col)

    # Height-augmented SI variables are tracked separately and appended later.
    if "ppg_si_m_s_approx" in base_features:
        for suffix in (
            "__N2", "__N3_minus_N2", "__REM_minus_N2", "__night_iqr"
        ):
            col = f"ppg_si_m_s_approx{suffix}"
            if col in wide.columns:
                candidate_columns.append(col)

    candidate_columns = list(dict.fromkeys(candidate_columns))
    return wide, candidate_columns, stage_median


# =============================================================================
# ABPM subject summary
# =============================================================================

def summarize_abpm(bp_linked: pd.DataFrame,
                   cfg: PhenotypeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bp = bp_linked.copy()
    required = ["subject_id", "night", "ABPM_exposure", "sbp", "dbp"]
    missing = [col for col in required if col not in bp.columns]
    if missing:
        raise KeyError(f"BP-linked table is missing required columns: {missing}")

    bp = bp.loc[normalize_bool(bp["ABPM_exposure"])].copy()
    for col in ["sbp", "dbp", "map", "bp_hr", "stage_purity", "stage_coverage"]:
        if col in bp.columns:
            bp[col] = pd.to_numeric(bp[col], errors="coerce")

    if "map" not in bp.columns:
        bp["map"] = bp["dbp"] + (bp["sbp"] - bp["dbp"]) / 3.0
    else:
        computed_map = bp["dbp"] + (bp["sbp"] - bp["dbp"]) / 3.0
        bp["map"] = bp["map"].fillna(computed_map)

    bp["pp"] = bp["sbp"] - bp["dbp"]

    # Primary nocturnal BP summary excludes Wake and requires reasonable stage
    # alignment. N1 is retained because ABPM sampling is sparse.
    if "dominant_stage" in bp.columns:
        bp = bp.loc[bp["dominant_stage"].isin(["N1", "N2", "N3", "REM"])].copy()
    if "stage_purity" in bp.columns:
        bp = bp.loc[
            bp["stage_purity"].ge(cfg.stage_purity_min)
            & bp["stage_coverage"].ge(cfg.stage_coverage_min)
        ].copy()

    bp = bp.loc[
        bp["sbp"].between(70, 250)
        & bp["dbp"].between(35, 150)
        & bp["map"].between(45, 190)
        & bp["pp"].between(10, 140)
    ].copy()

    # One row per cuff reading.
    dedup_cols = ["subject_id", "night"]
    if "bp_id" in bp.columns:
        dedup_cols.append("bp_id")
    elif "bp_timestamp" in bp.columns:
        dedup_cols.append("bp_timestamp")
    bp = bp.drop_duplicates(dedup_cols)

    sort_col = "bp_timestamp" if "bp_timestamp" in bp.columns else "bp_sec"
    if sort_col in bp.columns:
        bp[sort_col] = (
            pd.to_datetime(bp[sort_col], errors="coerce")
            if sort_col == "bp_timestamp"
            else pd.to_numeric(bp[sort_col], errors="coerce")
        )
        bp = bp.sort_values(["subject_id", "night", sort_col])

    rows: List[Dict[str, object]] = []
    for (subject_id, night), group in bp.groupby(
        ["subject_id", "night"], observed=True
    ):
        row: Dict[str, object] = {
            "subject_id": subject_id,
            "abpm_night": night,
            "abpm_n_readings": int(len(group)),
        }
        for variable in ("sbp", "dbp", "map", "pp"):
            x = pd.to_numeric(group[variable], errors="coerce").dropna()
            row[f"abpm_mean_{variable}"] = float(x.mean()) if len(x) else np.nan
            row[f"abpm_median_{variable}"] = float(x.median()) if len(x) else np.nan
            row[f"abpm_{variable}_sd"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
            row[f"abpm_{variable}_cv_pct"] = (
                float(x.std(ddof=1) / x.mean() * 100.0)
                if len(x) > 1 and abs(x.mean()) > 1e-12 else np.nan
            )
            row[f"abpm_{variable}_arv"] = average_real_variability(x)
            row[f"abpm_{variable}_p90"] = (
                float(np.percentile(x, 90)) if len(x) else np.nan
            )
            row[f"abpm_{variable}_range"] = (
                float(x.max() - x.min()) if len(x) else np.nan
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, bp

    summary["abpm_eligible"] = summary["abpm_n_readings"].ge(cfg.min_bp_readings)

    # Descriptive BP domain scores. Prediction models use raw targets.
    eligible = summary["abpm_eligible"]
    for output_col, components in {
        "abpm_level_composite_z": [
            "abpm_mean_map", "abpm_mean_pp",
        ],
        "abpm_variability_composite_z": [
            "abpm_map_arv", "abpm_sbp_sd",
        ],
    }.items():
        z_parts = []
        for component in components:
            z, _, _ = robust_z(summary.loc[eligible, component])
            full = pd.Series(np.nan, index=summary.index)
            full.loc[eligible] = z.to_numpy()
            z_parts.append(full)
        summary[output_col] = pd.concat(z_parts, axis=1).mean(axis=1)

    return summary, bp


# =============================================================================
# Repeatability statistics
# =============================================================================

def icc_two_way_single(a: Sequence[float],
                       b: Sequence[float]) -> Dict[str, float]:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(pair)
    k = 2
    result = {
        "n_pairs": n,
        "icc_a1": np.nan,
        "icc_c1": np.nan,
        "mean_difference_nonabpm_minus_abpm": np.nan,
        "loa_low": np.nan,
        "loa_high": np.nan,
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "ccc": np.nan,
    }
    if n < 3:
        return result

    data = pair[["a", "b"]].to_numpy(dtype=float)
    grand = data.mean()
    subject_means = data.mean(axis=1)
    night_means = data.mean(axis=0)

    ss_subject = k * np.sum((subject_means - grand) ** 2)
    ss_night = n * np.sum((night_means - grand) ** 2)
    residual = (
        data
        - subject_means[:, None]
        - night_means[None, :]
        + grand
    )
    ss_error = np.sum(residual ** 2)

    ms_subject = ss_subject / (n - 1)
    ms_night = ss_night / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    den_a = (
        ms_subject
        + (k - 1) * ms_error
        + k * (ms_night - ms_error) / n
    )
    den_c = ms_subject + (k - 1) * ms_error

    difference = data[:, 1] - data[:, 0]
    mean_diff = float(np.mean(difference))
    sd_diff = float(np.std(difference, ddof=1))

    result.update({
        "icc_a1": float((ms_subject - ms_error) / den_a)
        if abs(den_a) > 1e-12 else np.nan,
        "icc_c1": float((ms_subject - ms_error) / den_c)
        if abs(den_c) > 1e-12 else np.nan,
        "mean_difference_nonabpm_minus_abpm": mean_diff,
        "loa_low": mean_diff - 1.96 * sd_diff,
        "loa_high": mean_diff + 1.96 * sd_diff,
        "pearson_r": safe_corr(data[:, 0], data[:, 1], "pearson"),
        "spearman_r": safe_corr(data[:, 0], data[:, 1], "spearman"),
        "ccc": concordance_correlation_coefficient(data[:, 0], data[:, 1]),
    })
    return result


def bootstrap_icc_ci(a: Sequence[float],
                     b: Sequence[float],
                     n_bootstrap: int,
                     seed: int) -> Tuple[float, float]:
    pair = pd.DataFrame({"a": a, "b": b}).dropna().reset_index(drop=True)
    if len(pair) < 6 or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(pair), size=len(pair))
        value = icc_two_way_single(
            pair.loc[idx, "a"].to_numpy(),
            pair.loc[idx, "b"].to_numpy(),
        )["icc_a1"]
        if np.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(50, n_bootstrap // 10):
        return np.nan, np.nan
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def pair_nights(profile: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    abpm = (
        profile.loc[profile["ABPM_exposure"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    nonabpm = (
        profile.loc[~profile["ABPM_exposure"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    common = abpm.index.intersection(nonabpm.index)
    return abpm.loc[common].sort_index(), nonabpm.loc[common].sort_index()


def feature_repeatability_table(
    abpm_profile: pd.DataFrame,
    nonabpm_profile: pd.DataFrame,
    feature_columns: Sequence[str],
    cfg: PhenotypeConfig,
    bootstrap: bool = True,
) -> pd.DataFrame:
    rows = []
    for i, feature in enumerate(feature_columns):
        if feature not in abpm_profile.columns or feature not in nonabpm_profile.columns:
            continue
        stats_row = icc_two_way_single(
            abpm_profile[feature],
            nonabpm_profile[feature],
        )
        ci_low, ci_high = (np.nan, np.nan)
        if bootstrap:
            ci_low, ci_high = bootstrap_icc_ci(
                abpm_profile[feature],
                nonabpm_profile[feature],
                cfg.n_bootstrap,
                cfg.random_state + i,
            )
        stats_row.update({
            "feature": feature,
            "coverage_abpm": float(abpm_profile[feature].notna().mean()),
            "coverage_nonabpm": float(nonabpm_profile[feature].notna().mean()),
            "icc_a1_ci_low": ci_low,
            "icc_a1_ci_high": ci_high,
        })
        rows.append(stats_row)
    return pd.DataFrame(rows).sort_values(
        ["icc_a1", "n_pairs"], ascending=[False, False]
    )


# =============================================================================
# Same-person phenotype similarity
# =============================================================================

def choose_stable_signature_features(
    repeatability: pd.DataFrame,
    cfg: PhenotypeConfig,
    include_height: bool = False,
) -> List[str]:
    table = repeatability.copy()
    table = table.loc[
        table["n_pairs"].ge(cfg.min_repeatability_pairs)
        & table["coverage_abpm"].ge(cfg.min_feature_coverage)
        & table["coverage_nonabpm"].ge(cfg.min_feature_coverage)
    ].copy()

    if not include_height:
        table = table.loc[
            ~table["feature"].str.contains("ppg_si_m_s_approx", na=False)
        ]

    stable = table.loc[table["icc_a1"].ge(cfg.min_train_icc_a)].copy()
    if len(stable) < cfg.min_model_features:
        stable = table.sort_values(
            ["icc_a1", "n_pairs"], ascending=[False, False]
        ).head(cfg.min_model_features)

    return stable.sort_values(
        ["icc_a1", "n_pairs"], ascending=[False, False]
    )["feature"].head(cfg.max_model_features).tolist()


def standardize_signature(
    abpm_profile: pd.DataFrame,
    nonabpm_profile: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x1 = abpm_profile[list(features)].apply(pd.to_numeric, errors="coerce")
    x2 = nonabpm_profile[list(features)].apply(pd.to_numeric, errors="coerce")

    medians = x1.median()
    x1 = x1.fillna(medians)
    x2 = x2.fillna(medians)

    centers = x1.median()
    scales = (x1.quantile(0.75) - x1.quantile(0.25)) / 1.349
    fallback = x1.std(ddof=1)
    scales = scales.where(scales > 1e-12, fallback)
    scales = scales.where(scales > 1e-12, 1.0)

    z1 = ((x1 - centers) / scales).to_numpy(dtype=float)
    z2 = ((x2 - centers) / scales).to_numpy(dtype=float)

    params = pd.DataFrame({
        "feature": list(features),
        "center_abpm_median": centers.reindex(features).to_numpy(),
        "scale_abpm_robust_sd": scales.reindex(features).to_numpy(),
    })
    return z1, z2, params


def same_person_similarity_analysis(
    abpm_profile: pd.DataFrame,
    nonabpm_profile: pd.DataFrame,
    features: Sequence[str],
    cfg: PhenotypeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    if len(features) < 2:
        return pd.DataFrame(), pd.DataFrame(), np.empty((0, 0)), pd.DataFrame()

    z1, z2, params = standardize_signature(
        abpm_profile, nonabpm_profile, features
    )
    similarity = cosine_similarity_matrix(z1, z2)
    n = similarity.shape[0]
    subjects = abpm_profile.index.to_numpy()

    rows = []
    for i, subject_id in enumerate(subjects):
        row_sim = similarity[i]
        own = float(row_sim[i])
        order = np.argsort(-row_sim)
        rank = int(np.where(order == i)[0][0] + 1)
        best_idx = int(order[0])
        others = np.delete(row_sim, i)
        rows.append({
            "subject_id": subject_id,
            "own_cross_night_cosine": own,
            "mean_other_person_cosine": float(np.nanmean(others)),
            "own_minus_other_mean": own - float(np.nanmean(others)),
            "own_match_rank": rank,
            "top1_correct": bool(best_idx == i),
            "best_matching_subject": subjects[best_idx],
        })
    subject_table = pd.DataFrame(rows)

    observed_mean = float(subject_table["own_cross_night_cosine"].mean())
    observed_top1 = float(subject_table["top1_correct"].mean())

    rng = np.random.default_rng(cfg.random_state)
    perm_means = []
    perm_top1 = []
    for _ in range(cfg.n_permutations):
        permutation = rng.permutation(n)
        perm_diag = similarity[np.arange(n), permutation]
        perm_means.append(float(np.mean(perm_diag)))
        perm_top1.append(
            float(np.mean(permutation[np.argmax(similarity[:, permutation], axis=1)] == np.arange(n)))
        )

    summary = pd.DataFrame([{
        "n_subjects": n,
        "n_features": len(features),
        "mean_own_cosine": observed_mean,
        "mean_other_cosine": float(
            np.mean(similarity[~np.eye(n, dtype=bool)])
        ),
        "top1_identity_accuracy": observed_top1,
        "median_own_match_rank": float(subject_table["own_match_rank"].median()),
        "permutation_p_mean_own_similarity": (
            1.0 + np.sum(np.asarray(perm_means) >= observed_mean)
        ) / (cfg.n_permutations + 1.0),
        "permutation_p_top1_accuracy": (
            1.0 + np.sum(np.asarray(perm_top1) >= observed_top1)
        ) / (cfg.n_permutations + 1.0),
    }])

    return subject_table, summary, similarity, params


# =============================================================================
# Fold-specific feature selection and cross-fitted phenotype models
# =============================================================================

def correlation_prune(
    frame: pd.DataFrame,
    ordered_features: Sequence[str],
    threshold: float,
) -> List[str]:
    kept: List[str] = []
    x = frame[list(ordered_features)].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median())
    for feature in ordered_features:
        if feature not in x.columns or x[feature].nunique(dropna=True) < 2:
            continue
        accept = True
        for selected in kept:
            corr = x[[feature, selected]].corr().iloc[0, 1]
            if np.isfinite(corr) and abs(corr) >= threshold:
                accept = False
                break
        if accept:
            kept.append(feature)
    return kept


def select_features_within_training_fold(
    train_abpm: pd.DataFrame,
    train_nonabpm: pd.DataFrame,
    candidate_features: Sequence[str],
    cfg: PhenotypeConfig,
    include_height: bool,
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    for feature in candidate_features:
        if feature not in train_abpm.columns or feature not in train_nonabpm.columns:
            continue
        if not include_height and "ppg_si_m_s_approx" in feature:
            continue
        stats_row = icc_two_way_single(
            train_abpm[feature],
            train_nonabpm[feature],
        )
        stats_row.update({
            "feature": feature,
            "coverage_abpm": float(train_abpm[feature].notna().mean()),
            "coverage_nonabpm": float(train_nonabpm[feature].notna().mean()),
        })
        rows.append(stats_row)

    table = pd.DataFrame(rows)
    if table.empty:
        return [], table

    eligible = table.loc[
        table["n_pairs"].ge(
            min(cfg.min_repeatability_pairs, max(6, len(train_abpm) // 3))
        )
        & table["coverage_abpm"].ge(cfg.min_feature_coverage)
        & table["coverage_nonabpm"].ge(cfg.min_feature_coverage)
    ].copy()

    stable = eligible.loc[eligible["icc_a1"].ge(cfg.min_train_icc_a)].copy()
    stable = stable.sort_values(
        ["icc_a1", "n_pairs"], ascending=[False, False]
    )

    if len(stable) < cfg.min_model_features:
        stable = eligible.sort_values(
            ["icc_a1", "n_pairs"], ascending=[False, False]
        ).head(max(cfg.min_model_features, len(stable)))

    ordered = stable["feature"].head(cfg.max_model_features * 2).tolist()
    selected = correlation_prune(
        train_abpm,
        ordered,
        cfg.corr_prune_threshold,
    )[:cfg.max_model_features]

    # Final fallback: use the highest-coverage variables even when ICC is weak.
    if len(selected) < cfg.min_model_features:
        fallback = (
            table.assign(
                min_coverage=table[
                    ["coverage_abpm", "coverage_nonabpm"]
                ].min(axis=1)
            )
            .sort_values(
                ["min_coverage", "icc_a1", "n_pairs"],
                ascending=[False, False, False],
            )["feature"]
            .tolist()
        )
        selected = correlation_prune(
            train_abpm,
            list(dict.fromkeys(selected + fallback)),
            cfg.corr_prune_threshold,
        )[:cfg.max_model_features]

    return selected, table


def make_regression_pipeline(n_train: int,
                             n_inner_folds: int,
                             random_state: int) -> Pipeline:
    inner_splits = min(n_inner_folds, max(2, n_train // 6))
    inner_cv = KFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    estimator = ElasticNetCV(
        l1_ratio=[0.05, 0.20, 0.50, 0.80, 0.95],
        alphas=np.logspace(-3, 2, 50),
        cv=inner_cv,
        max_iter=50000,
        random_state=random_state,
        selection="cyclic",
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def fit_with_fallback(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: PhenotypeConfig,
    seed: int,
) -> Tuple[Pipeline, str]:
    primary = make_regression_pipeline(
        len(x_train), cfg.n_inner_folds, seed
    )
    try:
        primary.fit(x_train, y_train)
        return primary, "ElasticNetCV"
    except Exception as exc:
        warnings.warn(f"ElasticNetCV failed; using RidgeCV. Reason: {exc}")
        ridge = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 50))),
        ])
        ridge.fit(x_train, y_train)
        return ridge, "RidgeCV_fallback"


def crossfit_abpm_axis(
    abpm_profile: pd.DataFrame,
    nonabpm_profile: pd.DataFrame,
    target: pd.Series,
    candidate_features: Sequence[str],
    cfg: PhenotypeConfig,
    specification: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = (
        abpm_profile.index
        .intersection(nonabpm_profile.index)
        .intersection(target.dropna().index)
    )
    a = abpm_profile.loc[common].copy()
    n = nonabpm_profile.loc[common].copy()
    y = pd.to_numeric(target.loc[common], errors="coerce")

    valid_subjects = y.dropna().index
    a = a.loc[valid_subjects]
    n = n.loc[valid_subjects]
    y = y.loc[valid_subjects]

    n_subjects = len(y)
    if n_subjects < 12:
        raise ValueError(
            f"Too few subjects for cross-fitting: {n_subjects}"
        )

    n_splits = min(cfg.n_outer_folds, max(2, n_subjects // 10))
    outer = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    prediction_rows = []
    selection_rows = []
    subject_array = y.index.to_numpy()

    include_height = specification == "ppg_plus_height"

    for fold, (train_idx, test_idx) in enumerate(outer.split(subject_array), start=1):
        train_subjects = subject_array[train_idx]
        test_subjects = subject_array[test_idx]

        selected, selection_table = select_features_within_training_fold(
            a.loc[train_subjects],
            n.loc[train_subjects],
            candidate_features,
            cfg,
            include_height=include_height,
        )
        if len(selected) == 0:
            raise RuntimeError(f"No usable features selected in fold {fold}.")

        model, model_type = fit_with_fallback(
            a.loc[train_subjects, selected],
            y.loc[train_subjects],
            cfg,
            cfg.random_state + fold,
        )

        pred_abpm = model.predict(a.loc[test_subjects, selected])
        pred_nonabpm = model.predict(n.loc[test_subjects, selected])

        for subject_id, p1, p2 in zip(
            test_subjects, pred_abpm, pred_nonabpm
        ):
            prediction_rows.append({
                "subject_id": subject_id,
                "fold": fold,
                "specification": specification,
                "observed_target": float(y.loc[subject_id]),
                "predicted_from_abpm_night_ppg": float(p1),
                "predicted_from_nonabpm_night_ppg": float(p2),
                "model_type": model_type,
                "n_selected_features": len(selected),
            })

        selected_set = set(selected)
        for feature in candidate_features:
            if not include_height and "ppg_si_m_s_approx" in feature:
                continue
            repeat_row = selection_table.loc[
                selection_table["feature"].eq(feature)
            ]
            selection_rows.append({
                "fold": fold,
                "specification": specification,
                "feature": feature,
                "selected": feature in selected_set,
                "training_icc_a1": (
                    float(repeat_row["icc_a1"].iloc[0])
                    if len(repeat_row) else np.nan
                ),
                "training_n_pairs": (
                    int(repeat_row["n_pairs"].iloc[0])
                    if len(repeat_row) else 0
                ),
            })

    predictions = pd.DataFrame(prediction_rows).sort_values("subject_id")
    selections = pd.DataFrame(selection_rows)
    return predictions, selections


def evaluate_crossfit_predictions(
    predictions: pd.DataFrame,
    axis_name: str,
) -> pd.DataFrame:
    rows = []
    for specification, group in predictions.groupby(
        "specification", observed=True
    ):
        observed = group["observed_target"]
        p1 = group["predicted_from_abpm_night_ppg"]
        p2 = group["predicted_from_nonabpm_night_ppg"]

        repeat = icc_two_way_single(p1, p2)
        labels1 = tertile_labels(p1)
        labels2 = tertile_labels(p2, reference=p1)
        valid_labels = labels1.notna() & labels2.notna()

        rows.append({
            "axis": axis_name,
            "specification": specification,
            "n_subjects": len(group),
            "abpm_night_oof_mae": mean_absolute_error(observed, p1),
            "abpm_night_oof_r2": r2_score(observed, p1),
            "abpm_night_oof_pearson_r": safe_corr(observed, p1, "pearson"),
            "abpm_night_oof_spearman_r": safe_corr(observed, p1, "spearman"),
            "nonabpm_night_to_observed_mae": mean_absolute_error(observed, p2),
            "nonabpm_night_to_observed_r2": r2_score(observed, p2),
            "nonabpm_night_to_observed_pearson_r": safe_corr(
                observed, p2, "pearson"
            ),
            "nonabpm_night_to_observed_spearman_r": safe_corr(
                observed, p2, "spearman"
            ),
            "cross_night_prediction_icc_a1": repeat["icc_a1"],
            "cross_night_prediction_icc_c1": repeat["icc_c1"],
            "cross_night_prediction_ccc": repeat["ccc"],
            "cross_night_prediction_pearson_r": repeat["pearson_r"],
            "cross_night_prediction_mean_difference": repeat[
                "mean_difference_nonabpm_minus_abpm"
            ],
            "cross_night_prediction_loa_low": repeat["loa_low"],
            "cross_night_prediction_loa_high": repeat["loa_high"],
            "tertile_agreement": (
                float((labels1[valid_labels] == labels2[valid_labels]).mean())
                if valid_labels.any() else np.nan
            ),
            "tertile_weighted_kappa": (
                float(cohen_kappa_score(
                    labels1[valid_labels],
                    labels2[valid_labels],
                    weights="quadratic",
                ))
                if valid_labels.sum() >= 6 else np.nan
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Multidimensional predicted phenotype vector
# =============================================================================

def assemble_predicted_phenotype(
    all_predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide_parts = []
    for (axis, specification), group in all_predictions.groupby(
        ["axis", "specification"], observed=True
    ):
        part = group[[
            "subject_id",
            "observed_target",
            "predicted_from_abpm_night_ppg",
            "predicted_from_nonabpm_night_ppg",
        ]].copy()
        part = part.rename(columns={
            "observed_target": f"{axis}__observed",
            "predicted_from_abpm_night_ppg": f"{axis}__pred_abpm",
            "predicted_from_nonabpm_night_ppg": f"{axis}__pred_nonabpm",
        })
        part["specification"] = specification
        wide_parts.append(part)

    if not wide_parts:
        return pd.DataFrame(), pd.DataFrame()

    combined = None
    for part in wide_parts:
        if combined is None:
            combined = part
        else:
            combined = combined.merge(
                part,
                on=["subject_id", "specification"],
                how="outer",
            )

    similarity_rows = []
    for specification, group in combined.groupby(
        "specification", observed=True
    ):
        abpm_cols = sorted(
            [c for c in group.columns if c.endswith("__pred_abpm")]
        )
        axes = [c.replace("__pred_abpm", "") for c in abpm_cols]
        non_cols = [f"{axis}__pred_nonabpm" for axis in axes]

        valid_axes = [
            axis for axis in axes
            if group[f"{axis}__pred_abpm"].notna().sum() >= 10
            and group[f"{axis}__pred_nonabpm"].notna().sum() >= 10
        ]
        if len(valid_axes) < 2:
            continue

        z_abpm = []
        z_non = []
        for axis in valid_axes:
            p1 = group[f"{axis}__pred_abpm"]
            p2 = group[f"{axis}__pred_nonabpm"]
            z1, center, scale = robust_z(p1)
            z2, _, _ = robust_z(p2, center=center, scale=scale)
            z_abpm.append(z1.to_numpy())
            z_non.append(z2.to_numpy())

        matrix1 = np.column_stack(z_abpm)
        matrix2 = np.column_stack(z_non)
        valid_rows = (
            np.isfinite(matrix1).all(axis=1)
            & np.isfinite(matrix2).all(axis=1)
        )
        own_cosine = np.full(len(group), np.nan)
        own_cosine[valid_rows] = cosine_similarity_rows(
            matrix1[valid_rows], matrix2[valid_rows]
        )
        euclidean_change = np.full(len(group), np.nan)
        euclidean_change[valid_rows] = np.linalg.norm(
            matrix2[valid_rows] - matrix1[valid_rows], axis=1
        )

        for idx, (_, row) in enumerate(group.iterrows()):
            similarity_rows.append({
                "subject_id": row["subject_id"],
                "specification": specification,
                "n_axes": len(valid_axes),
                "phenotype_vector_cosine": own_cosine[idx],
                "phenotype_vector_euclidean_change": euclidean_change[idx],
            })

    return combined, pd.DataFrame(similarity_rows)


# =============================================================================
# Exploratory cluster transfer
# =============================================================================

def cluster_transfer_analysis(
    phenotype_table: pd.DataFrame,
    cfg: PhenotypeConfig,
) -> pd.DataFrame:
    if phenotype_table.empty or not cfg.run_cluster_transfer:
        return pd.DataFrame()

    rows = []
    for specification, group in phenotype_table.groupby(
        "specification", observed=True
    ):
        abpm_cols = sorted(
            [c for c in group.columns if c.endswith("__pred_abpm")]
        )
        axes = [c.replace("__pred_abpm", "") for c in abpm_cols]
        non_cols = [f"{axis}__pred_nonabpm" for axis in axes]
        if len(axes) < 2 or any(c not in group.columns for c in non_cols):
            continue

        x1 = group[abpm_cols].apply(pd.to_numeric, errors="coerce")
        x2 = group[non_cols].apply(pd.to_numeric, errors="coerce")
        valid = x1.notna().all(axis=1) & x2.notna().all(axis=1)
        x1 = x1.loc[valid]
        x2 = x2.loc[valid]
        if len(x1) < 20:
            continue

        scaler = StandardScaler().fit(x1)
        z1 = scaler.transform(x1)
        z2 = scaler.transform(x2)

        for k in cfg.cluster_k_values:
            if len(x1) <= k * 3:
                continue
            model = KMeans(
                n_clusters=k,
                n_init=100,
                random_state=cfg.random_state,
            ).fit(z1)
            labels1 = model.labels_
            labels2 = model.predict(z2)
            cluster_sizes = np.bincount(labels1, minlength=k)
            silhouette = (
                silhouette_score(z1, labels1)
                if len(np.unique(labels1)) > 1 else np.nan
            )
            acceptable = bool(
                np.min(cluster_sizes) >= cfg.min_cluster_size
                and np.isfinite(silhouette)
                and silhouette >= cfg.min_silhouette
            )
            rows.append({
                "specification": specification,
                "k": k,
                "n_subjects": len(x1),
                "silhouette_abpm_night": silhouette,
                "minimum_cluster_size": int(np.min(cluster_sizes)),
                "acceptable_cluster_structure": acceptable,
                "direct_cluster_agreement": float(np.mean(labels1 == labels2)),
                "cohen_kappa": float(cohen_kappa_score(labels1, labels2)),
                "adjusted_rand_index": float(
                    adjusted_rand_score(labels1, labels2)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(labels1, labels2)
                ),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def plot_feature_repeatability(
    repeatability: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    if repeatability.empty:
        return
    table = repeatability.dropna(subset=["icc_a1"]).head(top_n).copy()
    if table.empty:
        return
    table = table.sort_values("icc_a1")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(table))))
    y = np.arange(len(table))
    ax.errorbar(
        table["icc_a1"],
        y,
        xerr=np.vstack([
            table["icc_a1"] - table["icc_a1_ci_low"],
            table["icc_a1_ci_high"] - table["icc_a1"],
        ]),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.50, linestyle="--", linewidth=1)
    ax.axvline(0.75, linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel("ICC(A,1), ABPM night vs non-ABPM night")
    ax.set_title("Most repeatable PPG phenotype features")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_repeatability_forest.png", dpi=200)
    plt.close(fig)


def plot_crossfit_axis(
    predictions: pd.DataFrame,
    axis: str,
    output_dir: Path,
) -> None:
    for specification, group in predictions.groupby(
        "specification", observed=True
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            group["predicted_from_abpm_night_ppg"],
            group["predicted_from_nonabpm_night_ppg"],
            alpha=0.75,
        )
        values = np.concatenate([
            group["predicted_from_abpm_night_ppg"].to_numpy(),
            group["predicted_from_nonabpm_night_ppg"].to_numpy(),
        ])
        values = values[np.isfinite(values)]
        if len(values):
            low, high = np.min(values), np.max(values)
            ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("OOF phenotype score from ABPM-night PPG")
        ax.set_ylabel("Score from non-ABPM-night PPG")
        ax.set_title(f"{axis}: {specification}")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{axis}_{specification}_cross_night_scatter.png",
            dpi=200,
        )
        plt.close(fig)


def plot_similarity_distribution(
    subject_similarity: pd.DataFrame,
    output_dir: Path,
) -> None:
    if subject_similarity.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        subject_similarity["own_minus_other_mean"].dropna(),
        bins=20,
        alpha=0.8,
    )
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Own-person cosine minus mean other-person cosine")
    ax.set_ylabel("Participants")
    ax.set_title("Cross-night personal PPG signature specificity")
    fig.tight_layout()
    fig.savefig(output_dir / "same_person_similarity_margin.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Output guide
# =============================================================================

def write_readme(
    cfg: PhenotypeConfig,
    n_qc_windows: int,
    n_paired_subjects: int,
    selected_features: Sequence[str],
) -> None:
    text = f"""# ABPM-anchored PPG phenotype output

## Input

- Base extraction output: `{cfg.base_output_dir}`
- Phenotype output: `{cfg.output_dir}`
- QC-passed stable windows: **{n_qc_windows}**
- Participants with ABPM and non-ABPM PPG profiles: **{n_paired_subjects}**

## Main files

- `01_qc_night_inventory.csv`: retained PPG windows by subject-night.
- `02_stage_level_ppg_summary.csv`: stage-specific median PPG variables.
- `03_subject_night_ppg_profiles.csv`: wide subject-night PPG phenotype table.
- `04_abpm_subject_summary.csv`: nocturnal ABPM level and variability.
- `05_ppg_feature_repeatability.csv`: ICC(A,1), ICC(C,1), CCC and limits of agreement.
- `06_selected_personal_signature_features.csv`: variables used for descriptive identity matching.
- `07_same_person_similarity.csv`: whether each cuff-free night matches its own ABPM night.
- `08_same_person_similarity_summary.csv`: top-1 matching and permutation tests.
- `09_crossfit_predictions.csv`: strictly out-of-fold ABPM-anchored phenotype predictions.
- `10_crossfit_axis_performance.csv`: BP validity and cross-night score repeatability.
- `11_feature_selection_frequency.csv`: fold-specific feature selection stability.
- `12_subject_predicted_phenotype.csv`: multidimensional predicted phenotype axes.
- `13_phenotype_vector_similarity.csv`: cross-night vector-level similarity.
- `14_exploratory_cluster_transfer.csv`: fixed-centroid cluster transfer; use only when marked acceptable.

## Selected descriptive signature variables

{chr(10).join(f"- `{x}`" for x in selected_features)}

## Interpretation boundaries

1. The non-ABPM night does not validate BP repeatability.
2. Strict PPG-only models exclude SI because SI uses height.
3. AI is an explicit transform of RI and is not included simultaneously with RI.
4. Absolute PPG amplitude is not used across participants. Amplitude is expressed
   relative to the same night's stable N2 reference.
5. Cluster results are exploratory. Continuous phenotype axes remain primary.
6. APG a/b/c/d/e ratios require raw beat-level re-extraction and are not inferred
   from the current summary tables.
"""
    (cfg.output_dir / "README_RESULTS.md").write_text(text, encoding="utf-8")


# =============================================================================
# Main orchestration
# =============================================================================

def run_singapore_phenotype_pipeline(cfg: PhenotypeConfig) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.output_dir)
    figure_dir = ensure_dir(cfg.output_dir / "figures")

    config_payload = asdict(cfg)
    config_payload["base_output_dir"] = str(cfg.base_output_dir)
    config_payload["output_dir"] = str(cfg.output_dir)
    config_payload["demographic_file"] = (
        str(cfg.demographic_file) if cfg.demographic_file is not None else None
    )
    (cfg.output_dir / "00_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stable_raw, bp_raw, inventory_raw = load_inputs(cfg)
    demographics = load_demographics(cfg.demographic_file)

    stable_qc, qc_inventory = apply_ppg_qc(stable_raw, cfg)
    stable_qc = add_derived_ppg_features(stable_qc, demographics)

    subject_night_profile, candidate_features, stage_summary = (
        summarize_subject_night_ppg(stable_qc, cfg)
    )

    abpm_summary, bp_clean = summarize_abpm(bp_raw, cfg)

    # Keep paired nights and an eligible ABPM target.
    abpm_profile, nonabpm_profile = pair_nights(subject_night_profile)
    eligible_abpm = (
        abpm_summary.loc[abpm_summary["abpm_eligible"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    common = (
        abpm_profile.index
        .intersection(nonabpm_profile.index)
        .intersection(eligible_abpm.index)
    )
    abpm_profile = abpm_profile.loc[common].sort_index()
    nonabpm_profile = nonabpm_profile.loc[common].sort_index()
    eligible_abpm = eligible_abpm.loc[common].sort_index()

    # Save core tables before modeling.
    qc_inventory.to_csv(
        cfg.output_dir / "01_qc_night_inventory.csv", index=False
    )
    stage_summary.to_csv(
        cfg.output_dir / "02_stage_level_ppg_summary.csv", index=False
    )
    subject_night_profile.to_csv(
        cfg.output_dir / "03_subject_night_ppg_profiles.csv", index=False
    )
    abpm_summary.to_csv(
        cfg.output_dir / "04_abpm_subject_summary.csv", index=False
    )
    bp_clean.to_csv(
        cfg.output_dir / "04b_clean_nocturnal_bp_readings.csv", index=False
    )

    # Descriptive feature repeatability.
    repeatability = feature_repeatability_table(
        abpm_profile,
        nonabpm_profile,
        candidate_features,
        cfg,
        bootstrap=True,
    )
    repeatability.to_csv(
        cfg.output_dir / "05_ppg_feature_repeatability.csv", index=False
    )

    selected_signature = choose_stable_signature_features(
        repeatability, cfg, include_height=False
    )
    pd.DataFrame({"feature": selected_signature}).to_csv(
        cfg.output_dir / "06_selected_personal_signature_features.csv",
        index=False,
    )

    (
        subject_similarity,
        similarity_summary,
        similarity_matrix,
        scaling_params,
    ) = same_person_similarity_analysis(
        abpm_profile,
        nonabpm_profile,
        selected_signature,
        cfg,
    )
    subject_similarity.to_csv(
        cfg.output_dir / "07_same_person_similarity.csv", index=False
    )
    similarity_summary.to_csv(
        cfg.output_dir / "08_same_person_similarity_summary.csv", index=False
    )
    scaling_params.to_csv(
        cfg.output_dir / "08b_signature_scaling_parameters.csv", index=False
    )
    if similarity_matrix.size:
        pd.DataFrame(
            similarity_matrix,
            index=abpm_profile.index,
            columns=nonabpm_profile.index,
        ).to_csv(cfg.output_dir / "08c_cross_subject_similarity_matrix.csv")

    # Cross-fitted ABPM-anchored axes.
    prediction_tables = []
    selection_tables = []
    performance_tables = []

    available_specs = ["strict_ppg"]
    si_features = [
        c for c in candidate_features if "ppg_si_m_s_approx" in c
    ]
    if si_features and any(
        abpm_profile[c].notna().mean() >= cfg.min_feature_coverage
        for c in si_features
    ):
        available_specs.append("ppg_plus_height")

    for axis_name, target_col in BP_TARGETS.items():
        if target_col not in eligible_abpm.columns:
            warnings.warn(f"Target missing and skipped: {target_col}")
            continue
        target = eligible_abpm[target_col]

        for specification in available_specs:
            predictions, selections = crossfit_abpm_axis(
                abpm_profile,
                nonabpm_profile,
                target,
                candidate_features,
                cfg,
                specification=specification,
            )
            predictions["axis"] = axis_name
            selections["axis"] = axis_name
            prediction_tables.append(predictions)
            selection_tables.append(selections)

        axis_predictions = pd.concat(
            [p for p in prediction_tables if p["axis"].iloc[0] == axis_name],
            ignore_index=True,
        )
        performance_tables.append(
            evaluate_crossfit_predictions(axis_predictions, axis_name)
        )
        plot_crossfit_axis(axis_predictions, axis_name, figure_dir)

    all_predictions = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables else pd.DataFrame()
    )
    all_selections = (
        pd.concat(selection_tables, ignore_index=True)
        if selection_tables else pd.DataFrame()
    )
    performance = (
        pd.concat(performance_tables, ignore_index=True)
        if performance_tables else pd.DataFrame()
    )

    all_predictions.to_csv(
        cfg.output_dir / "09_crossfit_predictions.csv", index=False
    )
    performance.to_csv(
        cfg.output_dir / "10_crossfit_axis_performance.csv", index=False
    )

    if not all_selections.empty:
        selection_frequency = (
            all_selections.groupby(
                ["axis", "specification", "feature"], observed=True
            )
            .agg(
                selection_frequency=("selected", "mean"),
                selected_folds=("selected", "sum"),
                total_folds=("selected", "size"),
                median_training_icc_a1=("training_icc_a1", "median"),
                median_training_n_pairs=("training_n_pairs", "median"),
            )
            .reset_index()
            .sort_values(
                ["axis", "specification", "selection_frequency",
                 "median_training_icc_a1"],
                ascending=[True, True, False, False],
            )
        )
    else:
        selection_frequency = pd.DataFrame()
    selection_frequency.to_csv(
        cfg.output_dir / "11_feature_selection_frequency.csv", index=False
    )

    phenotype_table, vector_similarity = assemble_predicted_phenotype(
        all_predictions
    )
    phenotype_table.to_csv(
        cfg.output_dir / "12_subject_predicted_phenotype.csv", index=False
    )
    vector_similarity.to_csv(
        cfg.output_dir / "13_phenotype_vector_similarity.csv", index=False
    )

    cluster_transfer = cluster_transfer_analysis(phenotype_table, cfg)
    cluster_transfer.to_csv(
        cfg.output_dir / "14_exploratory_cluster_transfer.csv", index=False
    )

    # Figures
    plot_feature_repeatability(repeatability, figure_dir)
    plot_similarity_distribution(subject_similarity, figure_dir)

    write_readme(
        cfg,
        n_qc_windows=len(stable_qc),
        n_paired_subjects=len(common),
        selected_features=selected_signature,
    )

    print("=" * 72)
    print("Singapore ABPM-anchored PPG phenotype analysis complete")
    print(f"Output directory: {cfg.output_dir}")
    print(f"QC-passed windows: {len(stable_qc)}")
    print(f"Paired eligible subjects: {len(common)}")
    print(f"Descriptive signature features: {len(selected_signature)}")
    print("=" * 72)

    return {
        "stable_qc": stable_qc,
        "qc_inventory": qc_inventory,
        "stage_summary": stage_summary,
        "subject_night_profile": subject_night_profile,
        "abpm_summary": abpm_summary,
        "feature_repeatability": repeatability,
        "subject_similarity": subject_similarity,
        "similarity_summary": similarity_summary,
        "crossfit_predictions": all_predictions,
        "crossfit_performance": performance,
        "feature_selection_frequency": selection_frequency,
        "phenotype_table": phenotype_table,
        "vector_similarity": vector_similarity,
        "cluster_transfer": cluster_transfer,
    }


# =============================================================================
# Execute
# =============================================================================

if __name__ == "__main__":
    CONFIG = PhenotypeConfig(
        base_output_dir=BASE_OUTPUT_DIR,
        output_dir=PHENOTYPE_OUTPUT_DIR,
        demographic_file=DEMOGRAPHIC_FILE,
    )
    RESULTS = run_singapore_phenotype_pipeline(CONFIG)
