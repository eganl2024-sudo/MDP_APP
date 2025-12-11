"""
Coach View Metrics Engine for Player Analysis

Provides unified data contract and computation for Coach-facing metrics:
- Player-session metrics (intensity, load, MDP windows, trends, baselines)
- Team-level percentiles for context
- Insights generation from metrics
- Z-score normalized intensity scoring (percentile-based)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import date, timedelta
from intensity_utils import compute_mdp, percentile_rank, classify_intensity


# Note: compute_mdp is imported from intensity_utils (canonical source of truth)


def infer_sampling_frequency(series: pd.Series) -> float:
    """
    Infer sampling frequency (seconds per sample) from a time-indexed series.
    
    Args:
        series: Indexed Series (RangeIndex assumed 1Hz, or datetime index)
    
    Returns:
        Seconds per sample
    """
    if isinstance(series.index, pd.DatetimeIndex):
        # Compute median time difference
        diffs = series.index.to_series().diff().dropna()
        if len(diffs) > 0:
            median_diff = diffs.median()
            return median_diff.total_seconds()
    
    # Default: 1Hz (1 sample per second)
    return 1.0


def compute_mp_metrics(mp_series: pd.Series) -> Dict[str, float]:
    """
    Compute all MP-based metrics for a player session.
    
    Args:
        mp_series: Time series of metabolic power (W/kg)
    
    Returns:
        Dict with mdp10, mdp20, mdp30, total_load, early_mp, mid_mp, late_mp
    """
    if mp_series is None or mp_series.empty:
        return {
            'mdp10': np.nan,
            'mdp20': np.nan,
            'mdp30': np.nan,
            'total_load': np.nan,
            'early_mp': np.nan,
            'mid_mp': np.nan,
            'late_mp': np.nan
        }
    
    # Infer frequency
    freq_seconds = infer_sampling_frequency(mp_series)
    
    # Use canonical compute_mdp function (converts seconds to samples internally)
    mdp10, _ = compute_mdp(mp_series, 10)
    mdp20, _ = compute_mdp(mp_series, 20)
    mdp30, _ = compute_mdp(mp_series, 30)
    
    # Total load: sum of MP (assuming MP is already per sample)
    dt = freq_seconds
    total_load = float((mp_series * dt).sum()) if len(mp_series) > 0 else np.nan
    
    # Early / Mid / Late thirds
    n = len(mp_series)
    if n == 0:
        early_mp = mid_mp = late_mp = np.nan
    else:
        one_third = n // 3
        early_slice = mp_series.iloc[:one_third]
        mid_slice = mp_series.iloc[one_third:2 * one_third]
        late_slice = mp_series.iloc[2 * one_third:]
        
        early_mp = float(early_slice.mean()) if not early_slice.empty else np.nan
        mid_mp = float(mid_slice.mean()) if not mid_slice.empty else np.nan
        late_mp = float(late_slice.mean()) if not late_slice.empty else np.nan
    
    return {
        'mdp10': mdp10,
        'mdp20': mdp20,
        'mdp30': mdp30,
        'total_load': total_load,
        'early_mp': early_mp,
        'mid_mp': mid_mp,
        'late_mp': late_mp
    }


def get_player_mp_series(
    session_data: pd.DataFrame,
    player_id: str,
    mp_column: str = 'mp'
) -> Optional[pd.Series]:
    """
    Extract MP series for a player from session data.
    
    Args:
        session_data: DataFrame with all players
        player_id: Target player
        mp_column: Column name for MP values
    
    Returns:
        Series of MP values, or None if not found
    """
    player_data = session_data[session_data['player_id'] == player_id]
    
    if player_data.empty:
        return None
    
    # Sort by timestamp if available
    if 'timestamp' in player_data.columns:
        player_data = player_data.sort_values('timestamp')
    
    if mp_column not in player_data.columns:
        return None
    
    return player_data[mp_column].dropna()


# ============================================================================
# INTENSITY SCORING
# ============================================================================

def compute_intensity_score(row: pd.Series) -> float:
    """
    Map metrics to a 0-100 intensity score.
    
    Simple heuristic: average of normalized MDP10 and total load.
    Can be extended with z-score normalization if baseline data available.
    
    Args:
        row: Metrics row with mdp10, total_load, etc.
    
    Returns:
        Intensity score 0-100, or np.nan if insufficient data
    """
    scores = []
    
    # Normalize MDP10 (assume typical max is 40 W/kg, min is 5)
    if pd.notna(row.get('mdp10')):
        mdp10_norm = np.clip((row['mdp10'] - 5) / (40 - 5) * 100, 0, 100)
        scores.append(mdp10_norm)
    
    # Normalize total load (assume typical max is 5000 AU, min is 500)
    if pd.notna(row.get('total_load')):
        load_norm = np.clip((row['total_load'] - 500) / (5000 - 500) * 100, 0, 100)
        scores.append(load_norm)
    
    # Average the normalized scores
    if scores:
        return float(np.mean(scores))
    
    return np.nan


def get_player_28d_baseline(
    player_id: str,
    session_date: date,
    all_player_metrics_by_date: Dict[str, pd.DataFrame]
) -> float:
    """
    Compute 28-day rolling baseline intensity for a player.
    
    Args:
        player_id: Player ID
        session_date: Current session date
        all_player_metrics_by_date: Dict mapping dates to player metrics DataFrames
    
    Returns:
        Mean intensity score over past 28 days, or np.nan if no history
    """
    # Look back 28 days from session_date (exclusive)
    cutoff_date = session_date - timedelta(days=28)
    relevant_dates = [
        d for d in all_player_metrics_by_date.keys()
        if cutoff_date < d < session_date
    ]
    
    if not relevant_dates:
        return np.nan
    
    intensities = []
    for d in relevant_dates:
        metrics_df = all_player_metrics_by_date[d]
        player_row = metrics_df[metrics_df.get('player_id') == player_id]
        if not player_row.empty and pd.notna(player_row.iloc[0].get('intensity_score')):
            intensities.append(player_row.iloc[0]['intensity_score'])
    
    if intensities:
        return float(np.mean(intensities))
    
    return np.nan


# ============================================================================
# PERCENTILE COMPUTATION
# ============================================================================

def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank (0-100) for values in a series, handling NaNs.
    
    Args:
        series: Input series with potentially NaN values
    
    Returns:
        Series with percentile ranks (or NaN for original NaNs)
    """
    # Use canonical function from intensity_utils
    from intensity_utils import percentile_rank as _percentile_rank
    return _percentile_rank(series)


def add_session_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-session percentile columns to metrics DataFrame.
    
    Args:
        df: DataFrame with intensity_score, mdp10, mdp20, mdp30, total_load
    
    Returns:
        DataFrame with added percentile columns
    """
    df = df.copy()
    
    # Compute percentiles based on raw_intensity (if available) or intensity_score
    intensity_col = 'raw_intensity' if 'raw_intensity' in df.columns else 'intensity_score'
    df['intensity_percentile'] = percentile_rank(df[intensity_col])
    
    df['peak10_percentile'] = percentile_rank(df['mdp10'])
    df['peak20_percentile'] = percentile_rank(df['mdp20'])
    df['peak30_percentile'] = percentile_rank(df['mdp30'])
    df['total_load_percentile'] = percentile_rank(df['total_load'])
    
    return df


def compute_z_intensity_scores(
    raw_intensity_series: pd.Series,
    reference_intensities: pd.Series
) -> tuple:
    """
    Compute z-scores and percentiles using a reference dataset.
    
    Args:
        raw_intensity_series: Current session intensities
        reference_intensities: Series of intensities from reference period (e.g., last 6-8 weeks)
    
    Returns:
        Tuple of (z_scores Series, percentiles Series)
    """
    if reference_intensities is None or reference_intensities.empty:
        return (
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float),
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float)
        )
    
    valid_ref = reference_intensities.dropna()
    
    if valid_ref.empty or len(valid_ref) < 2:
        # Insufficient reference data
        return (
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float),
            pd.Series(np.nan, index=raw_intensity_series.index, dtype=float)
        )
    
    mean_ref = valid_ref.mean()
    std_ref = valid_ref.std(ddof=0)
    
    # Compute z-scores
    if std_ref == 0:
        z_scores = pd.Series(0.0, index=raw_intensity_series.index, dtype=float)
    else:
        z_scores = (raw_intensity_series - mean_ref) / std_ref
    
    # Compute percentiles relative to reference
    combined_series = pd.concat([raw_intensity_series, reference_intensities])
    combined_percentiles = percentile_rank(combined_series)
    z_percentiles = combined_percentiles.loc[raw_intensity_series.index]
    
    return z_scores, z_percentiles


# ============================================================================
# DATA CONTRACT: UNIFIED METRICS DATAFRAME
# ============================================================================

def get_player_session_metrics(
    session_data: pd.DataFrame,
    session_date: Optional[date] = None,
    all_player_metrics_by_date: Optional[Dict[str, pd.DataFrame]] = None,
    reference_intensities: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Return one row per player for the session with all intensity metrics.
    
    REQUIRED COLUMNS in session_data:
        player_id, player_number (optional), mp, timestamp
        (or intensity_10s, intensity_20s, intensity_30s columns already computed)
    
    RETURNS columns:
        player_id
        player_name (player_number if available, else player_id)
        intensity_score (0-100 composite score) [DEPRECATED: use intensity_percentile]
        raw_intensity (underlying intensity metric for z-score computation)
        z_intensity (standardized score)
        intensity_percentile (0-100 percentile rank)
        intensity_category (categorical label: Light, Moderate, Hard, Very hard)
        total_load (AU, sum of MP over session)
        mdp10, mdp20, mdp30 (W/kg, peak rolling averages)
        early_mp, mid_mp, late_mp (W/kg, thirds)
        baseline_28d_intensity (player's 28-day avg, or NaN)
        
        + percentiles:
        intensity_percentile, peak10_percentile, peak20_percentile,
        peak30_percentile, total_load_percentile
    
    Args:
        session_data: DataFrame with all players for one session
        session_date: Optional date for 28-day baseline lookup
        all_player_metrics_by_date: Optional dict for baseline computation
        reference_intensities: Optional Series of reference raw_intensity values for z-score
    
    Returns:
        DataFrame with one row per player, all columns populated
    """
    if session_data is None or session_data.empty:
        return pd.DataFrame()
    
    rows = []
    
    for player_id in session_data['player_id'].unique():
        # Extract MP series
        mp_series = get_player_mp_series(session_data, player_id, mp_column='mp')
        
        if mp_series is None or mp_series.empty:
            # Create minimal row with NaNs
            row = {
                'player_id': player_id,
                'player_name': f"Player {session_data[session_data['player_id'] == player_id]['player_number'].iloc[0]}" if 'player_number' in session_data.columns else str(player_id),
                'intensity_score': np.nan,  # Deprecated
                'raw_intensity': np.nan,
                'z_intensity': np.nan,
                'intensity_percentile': np.nan,
                'intensity_category': 'Unknown',
                'total_load': np.nan,
                'mdp10': np.nan,
                'mdp20': np.nan,
                'mdp30': np.nan,
                'early_mp': np.nan,
                'mid_mp': np.nan,
                'late_mp': np.nan,
                'baseline_28d_intensity': np.nan
            }
        else:
            # Compute all MP metrics
            mp_metrics = compute_mp_metrics(mp_series)
            
            # Compute raw intensity (legacy, for backward compat and as basis for z-score)
            raw_intensity = compute_intensity_score(mp_metrics)
            
            # Compute 28-day baseline if data provided
            baseline = np.nan
            if session_date and all_player_metrics_by_date:
                baseline = get_player_28d_baseline(player_id, session_date, all_player_metrics_by_date)
            
            # Build row
            row = {
                'player_id': player_id,
                'player_name': f"Player {session_data[session_data['player_id'] == player_id]['player_number'].iloc[0]}" if 'player_number' in session_data.columns else str(player_id),
                'intensity_score': raw_intensity,  # Deprecated: kept for backward compat
                'raw_intensity': raw_intensity,  # True underlying value
                'z_intensity': np.nan,  # Will be computed in add_session_percentiles
                'intensity_percentile': np.nan,  # Will be computed in add_session_percentiles
                'intensity_category': 'Unknown',  # Will be computed in add_session_percentiles
                'total_load': mp_metrics['total_load'],
                'mdp10': mp_metrics['mdp10'],
                'mdp20': mp_metrics['mdp20'],
                'mdp30': mp_metrics['mdp30'],
                'early_mp': mp_metrics['early_mp'],
                'mid_mp': mp_metrics['mid_mp'],
                'late_mp': mp_metrics['late_mp'],
                'baseline_28d_intensity': baseline
            }
        
        rows.append(row)
    
    metrics_df = pd.DataFrame(rows)
    
    # Add percentiles and z-scores
    metrics_df = add_session_percentiles(metrics_df)
    
    # Add z-scores if reference intensities provided
    if reference_intensities is not None and not reference_intensities.empty:
        z_scores, percentiles = compute_z_intensity_scores(
            metrics_df['raw_intensity'],
            reference_intensities
        )
        metrics_df['z_intensity'] = z_scores
        metrics_df['intensity_percentile'] = percentiles
    
    # Compute intensity category from percentile
    metrics_df['intensity_category'] = metrics_df['intensity_percentile'].apply(classify_intensity)
    
    return metrics_df


# ============================================================================
# INSIGHTS GENERATION
# ============================================================================

def ordinal_suffix(n: int) -> str:
    """Convert integer to ordinal suffix (1st, 2nd, 3rd, 4th, etc.)."""
    if n % 100 in (11, 12, 13):
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def generate_coach_insights(row: pd.Series) -> List[str]:
    """
    Generate rule-based coach insights from player metrics.
    
    Args:
        row: Metrics row with intensity_score, mdp10, baseline_28d_intensity, etc.
    
    Returns:
        List of insight strings (1-4 insights, or default if no data)
    """
    insights = []
    
    # Validate data
    if pd.isna(row.get('intensity_score')):
        return ["Insufficient data to generate insights."]
    
    intensity = row['intensity_score']
    mdp10 = row.get('mdp10')
    mdp20 = row.get('mdp20')
    total_load = row.get('total_load')
    baseline = row.get('baseline_28d_intensity')
    intensity_pct = row.get('intensity_percentile')
    peak10_pct = row.get('peak10_percentile')
    
    # Rule 1: High intensity vs baseline + high peak
    if pd.notna(baseline):
        delta_from_baseline = intensity - baseline
        if delta_from_baseline > 10 and pd.notna(peak10_pct) and peak10_pct > 75:
            insights.append(
                f"This session pushed this player above their normal intensity (+{delta_from_baseline:.0f} vs baseline) "
                f"with a very demanding worst-case period. Consider extra recovery."
            )
        elif delta_from_baseline < -10:
            insights.append(
                f"Below this player's typical intensity (−{abs(delta_from_baseline):.0f} vs baseline). "
                f"Good candidate for a lighter day or recovery session."
            )
    
    # Rule 2: High total load but moderate peak
    if pd.notna(total_load) and pd.notna(peak10_pct):
        if total_load > 3000 and peak10_pct < 50:
            insights.append(
                f"High volume day with relatively moderate worst-case demands. "
                f"Sustained effort with manageable peaks."
            )
    
    # Rule 3: Team percentile context
    if pd.notna(intensity_pct):
        if intensity_pct > 80:
            insights.append(
                f"One of the most demanding sessions on record for this player ({int(intensity_pct)}th percentile intensity). "
                f"Monitor for fatigue."
            )
        elif intensity_pct < 20:
            insights.append(
                f"Among the lighter sessions for this player ({int(intensity_pct)}th percentile). "
                f"Appropriate for recovery or technique work."
            )
    
    # Rule 4: Peak window comparison
    if pd.notna(mdp10) and pd.notna(mdp20):
        ratio = mdp10 / mdp20 if mdp20 > 0 else 1.0
        if ratio > 1.1:
            insights.append(
                f"Highest demands concentrated in short bursts (10s vs 20s windows). "
                f"Session included explosive efforts."
            )
    
    # Fallback
    if not insights:
        insights.append("Solid, within-normal session relative to this player's recent history.")
    
    return insights[:4]  # Cap at 4 insights
