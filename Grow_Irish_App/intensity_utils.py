"""
Intensity Utilities - Canonical MDP and Z-Score Intensity Computation

Shared utilities for both Coach and Analyst views:
- MDP (Most Demanding Period) computation for any window length
- Z-score based intensity percentile ranking
- Intensity category classification
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# CANONICAL MDP COMPUTATION (SINGLE SOURCE OF TRUTH)
# ============================================================================

def compute_mdp(mp: pd.Series, window_seconds: int) -> Tuple[float, Optional[int]]:
    """
    Compute peak average MP over any window_seconds period.
    
    Single source of truth for MDP calculation used by both Coach and Analyst views.
    
    Args:
        mp: Series of metabolic power values (W/kg)
        window_seconds: Window length in seconds (e.g., 10, 20, 30)
    
    Returns:
        Tuple of (peak_value_w_per_kg, peak_start_index)
        - peak_value: Highest rolling average, or np.nan if insufficient data
        - peak_start_index: Index position of peak window start, or None
    """
    if mp is None or mp.empty or window_seconds <= 0:
        return np.nan, None
    
    # Infer sampling interval (seconds per sample)
    if isinstance(mp.index, pd.DatetimeIndex):
        if len(mp) > 1:
            dt = (mp.index[1] - mp.index[0]).total_seconds()
        else:
            dt = 1.0  # Default fallback
    else:
        dt = 1.0  # Default 1 Hz for integer/range index
    
    # Convert window seconds to sample count
    window_samples = max(1, int(window_seconds / dt))
    
    # Compute rolling average
    rolling_mean = mp.rolling(window_samples, min_periods=window_samples).mean()
    
    # Find peak
    peak_value = rolling_mean.max()
    if pd.isna(peak_value):
        return np.nan, None
    
    peak_start_idx = rolling_mean.idxmax()
    return float(peak_value), peak_start_idx


# ============================================================================
# Z-SCORE INTENSITY NORMALIZATION (NEW)
# ============================================================================

def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank (0-100) for values in a series, handling NaNs.
    
    Args:
        series: Input series with potentially NaN values
    
    Returns:
        Series with percentile ranks (0-100), preserving original index
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    
    ranks = valid.rank(pct=True) * 100.0
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[valid.index] = ranks
    
    return out


def compute_z_scores_and_percentiles(
    raw_intensity_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute z-scores and percentiles from raw intensity values.
    
    Args:
        raw_intensity_series: Series of raw intensity values (e.g., from all sessions)
    
    Returns:
        Tuple of (z_scores, percentiles) as Series
    """
    valid = raw_intensity_series.dropna()
    
    if valid.empty or len(valid) < 2:
        # Insufficient data
        return (
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float),
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float)
        )
    
    mean_val = valid.mean()
    std_val = valid.std(ddof=0)
    
    # Handle zero std
    if std_val == 0:
        z_scores = pd.Series(np.nan, index=raw_intensity_series.index, dtype=float)
        z_scores.loc[valid.index] = 0.0
    else:
        z_scores = (raw_intensity_series - mean_val) / std_val
    
    # Percentile rank
    percentiles = percentile_rank(raw_intensity_series)
    
    return z_scores, percentiles


def classify_intensity(percentile: Optional[float]) -> str:
    """
    Map intensity percentile to categorical label.
    
    Args:
        percentile: Value 0-100, or NaN/None
    
    Returns:
        Category label: "Light", "Moderate", "Hard", "Very hard", or "Unknown"
    """
    if percentile is None or pd.isna(percentile):
        return "Unknown"
    
    if percentile < 25:
        return "Light"
    elif percentile < 60:
        return "Moderate"
    elif percentile < 85:
        return "Hard"
    else:
        return "Very hard"


# ============================================================================
# REFERENCE DATASET STATISTICS (FOR Z-SCORE BASELINE)
# ============================================================================

def compute_team_reference_stats(
    reference_df: pd.DataFrame,
    raw_intensity_column: str = 'raw_intensity'
) -> dict:
    """
    Compute team-wide mean, std, and percentile thresholds for z-score normalization.
    
    Args:
        reference_df: DataFrame with all reference sessions (e.g., last 6-8 weeks)
        raw_intensity_column: Column name containing raw intensity values
    
    Returns:
        Dict with:
        - 'mean': Mean intensity
        - 'std': Standard deviation
        - 'min': Minimum value
        - 'max': Maximum value
        - 'n_sessions': Sample count
        - 'percentile_25': 25th percentile value
        - 'percentile_50': 50th percentile value
        - 'percentile_75': 75th percentile value
    """
    if reference_df is None or reference_df.empty or raw_intensity_column not in reference_df.columns:
        return {}
    
    series = reference_df[raw_intensity_column].dropna()
    
    if series.empty:
        return {}
    
    return {
        'mean': float(series.mean()),
        'std': float(series.std(ddof=0)),
        'min': float(series.min()),
        'max': float(series.max()),
        'n_sessions': len(series),
        'percentile_25': float(series.quantile(0.25)),
        'percentile_50': float(series.quantile(0.50)),
        'percentile_75': float(series.quantile(0.75))
    }


# ============================================================================
# FORMATTING & DISPLAY HELPERS
# ============================================================================

def format_mdp_display(mdp_value: Optional[float]) -> str:
    """Format MDP value for display."""
    if mdp_value is None or pd.isna(mdp_value):
        return "N/A"
    return f"{mdp_value:.1f} W/kg"


def format_z_score(z: Optional[float]) -> str:
    """Format z-score for display."""
    if z is None or pd.isna(z):
        return "N/A"
    return f"z = {z:+.2f}"


def format_percentile(percentile: Optional[float]) -> str:
    """Format percentile with ordinal suffix."""
    if percentile is None or pd.isna(percentile):
        return "N/A"
    
    p = int(percentile)
    if p % 100 in (11, 12, 13):
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(p % 10, 'th')
    
    return f"{p}{suffix}"
