"""
MP-Based MDP + Session Intensity Index Pipeline

Computes metabolic power (MP) from tracking data and builds a session-level intensity index
based on peak power demands and volume metrics.

Input: Long-form tracking DataFrame with speed, acceleration, cadence per sample
Output: Session-level DataFrame with MDP metrics and session intensity scores
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IntensityWeights:
    """Weights for computing session intensity index."""
    w_explosiveness: float = 0.30
    w_repeatability: float = 0.50
    w_volume: float = 0.20


# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def clean_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw tracking data:
    - Drop unnamed columns and lat/long if present
    - Convert timestamp to datetime (from epoch ms or ISO string)
    - Add timestamp_s column (seconds since epoch as float)
    - Convert date to datetime format
    - Sort by player_id, event, timestamp
    """
    df = df.copy()
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Drop latitude/longitude if present
    df = df.drop(columns=[col for col in ['latitude', 'longitude'] if col in df.columns])
    
    # Convert timestamp to datetime
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add timestamp_s (seconds since epoch as float)
    df['timestamp_s'] = df['timestamp'].astype(np.int64) / 1e9
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by player_id, event, timestamp
    df = df.sort_values(['player_id', 'event', 'timestamp']).reset_index(drop=True)
    
    return df


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create session_id from player_id and event."""
    df = df.copy()
    df['session_id'] = df['player_id'].astype(str) + '_' + df['event'].astype(str)
    return df


# ============================================================================
# MP CALCULATION
# ============================================================================

def compute_mp_from_equation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metabolic power (MP) from the regression equation:
    
    MP = 1.990837
         + 26.695416 * speed
         - 0.184312 * speed^2
         - 14.958313 * a_pos
         - 21.754809 * a_neg
         + 23.417545 * (a_pos * speed)
         - 2.562471 * (a_neg * speed)
         - 0.017428 * cadence
    
    where a_pos = max(acc, 0), a_neg = min(acc, 0)
    
    Clamps MP to be non-negative.
    """
    df = df.copy()
    
    # Extract features
    speed = df['speed'].fillna(0)
    acc = df['acc'].fillna(0)
    cadence = df['cadence'].fillna(0)
    
    # Compute a_pos and a_neg
    a_pos = np.maximum(acc, 0)
    a_neg = np.minimum(acc, 0)
    
    # Apply MP equation
    mp = (
        1.990837
        + 26.695416 * speed
        - 0.184312 * (speed ** 2)
        - 14.958313 * a_pos
        - 21.754809 * a_neg
        + 23.417545 * (a_pos * speed)
        - 2.562471 * (a_neg * speed)
        - 0.017428 * cadence
    )
    
    # Clamp to non-negative
    df['mp_eq'] = np.maximum(mp, 0)
    
    return df


# ============================================================================
# PER-SESSION METRICS
# ============================================================================

def _estimate_sampling_interval_seconds(df_session: pd.DataFrame) -> float:
    """Estimate sampling interval (dt) from median timestamp differences."""
    if len(df_session) < 2:
        return 1.0
    
    diffs = df_session['timestamp_s'].diff().dropna()
    if len(diffs) == 0:
        return 1.0
    
    return float(diffs.median())

def _compute_window_mdp_for_session(
    df_session: pd.DataFrame,
    window_seconds_list: tuple = (10, 20, 30),
    mp_col: str = "mp_eq",
) -> dict:
    """
    Compute peak mean power demand (MDP) for each window duration.

    For each window W:
    - Estimate sample count per window = round(W / dt), minimum 1
    - Apply rolling mean with that window size
    - MDP_W = maximum of the rolling mean
    """
    result: dict = {}

    # No data at all for this player/session
    if df_session.empty:
        return {f"mdp_{int(w)}": np.nan for w in window_seconds_list}

    dt = _estimate_sampling_interval_seconds(df_session)
    mp_series = df_session[mp_col].fillna(0)

    # Guard against bad dt values (0, negative, NaN, inf)
    if not np.isfinite(dt) or dt <= 0:
        # Mark MDPs as unavailable for this broken session
        return {f"mdp_{int(w)}": np.nan for w in window_seconds_list}

        # If you prefer a fallback instead of NaNs, use:
        # dt = 1.0

    for window_sec in window_seconds_list:
        window_samples = max(1, int(round(window_sec / dt)))
        rolling_mean = mp_series.rolling(window=window_samples, min_periods=1).mean()
        mdp_value = float(rolling_mean.max())
        result[f"mdp_{int(window_sec)}"] = mdp_value

    return result



def _compute_session_mp_metrics(df_session: pd.DataFrame, mp_col: str = 'mp_eq') -> dict:
    """Compute session-level MP metrics: duration, mean MP, total load."""
    mp_series = df_session[mp_col].fillna(0)
    
    session_duration_s = float(df_session['timestamp_s'].max() - df_session['timestamp_s'].min())
    mean_mp = float(mp_series.mean())
    total_mp_load = mean_mp * session_duration_s if session_duration_s > 0 else 0.0
    
    return {
        'session_duration_s': session_duration_s,
        'mean_mp': mean_mp,
        'total_mp_load': total_mp_load
    }


def build_session_summary_df(
    df: pd.DataFrame,
    mp_col: str = 'mp_eq',
    window_seconds_list: tuple = (10, 20, 30)
) -> pd.DataFrame:
    """
    Build session-level summary DataFrame.
    
    Groups by player_id, session_id, date and computes:
    - MP metrics (duration, mean, total load)
    - MDP windows (10, 20, 30 seconds)
    - MDP peak value and window (max of 10/20/30)
    - Sessions per player
    - Days since last session
    """
    rows = []
    
    for (player_id, session_id, date), group in df.groupby(['player_id', 'session_id', 'date']):
        metrics = _compute_session_mp_metrics(group, mp_col=mp_col)
        mdp_metrics = _compute_window_mdp_for_session(group, window_seconds_list=window_seconds_list, mp_col=mp_col)
        
        # Compute MDP peak value and window
        mdp_10 = mdp_metrics.get('mdp_10', np.nan)
        mdp_20 = mdp_metrics.get('mdp_20', np.nan)
        mdp_30 = mdp_metrics.get('mdp_30', np.nan)
        
        mdp_values = {
            '10s': mdp_10,
            '20s': mdp_20,
            '30s': mdp_30,
        }
        
        # Find peak window, ignoring NaNs safely
        mdp_peak_window = max(
            mdp_values,
            key=lambda k: mdp_values[k] if pd.notna(mdp_values[k]) else -np.inf
        )
        mdp_peak_value = mdp_values[mdp_peak_window]
        
        row = {
            'player_id': player_id,
            'session_id': session_id,
            'date': date,
            **metrics,
            **mdp_metrics,
            'mdp_peak_value': mdp_peak_value,
            'mdp_peak_window': mdp_peak_window,
        }
        rows.append(row)
    
    session_df = pd.DataFrame(rows)
    session_df = session_df.sort_values(['player_id', 'date']).reset_index(drop=True)
    
    # Add sessions_per_player
    session_df['sessions_per_player'] = (
        session_df.groupby('player_id')['session_id'].transform('nunique')
    )
    
    # Add days_since_last_session
    session_df['days_since_last_session'] = (
        session_df.groupby('player_id')['date'].transform('diff').dt.days
    )
    
    return session_df


# ============================================================================
# COMPONENT SCORES & SESSION INTENSITY INDEX
# ============================================================================

def _z_score(series: pd.Series) -> pd.Series:
    """Compute z-scores; return zeros if std == 0."""
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mean) / std


def add_intensity_components(
    session_df: pd.DataFrame,
    weights: Optional[IntensityWeights] = None
) -> pd.DataFrame:
    """
    Add raw component scores and z-normalized scores.
    Compute session intensity index as weighted combination.
    """
    if weights is None:
        weights = IntensityWeights()
    
    df = session_df.copy()
    
    # Compute raw components
    df['explosiveness_raw'] = df['mdp_10']
    df['repeatability_raw'] = 0.5 * (df['mdp_20'] + df['mdp_30'])
    df['volume_raw'] = df['total_mp_load']
    
    # Compute z-scores
    df['explosiveness_z'] = _z_score(df['explosiveness_raw'])
    df['repeatability_z'] = _z_score(df['repeatability_raw'])
    df['volume_z'] = _z_score(df['volume_raw'])
    
    # Compute session intensity index
    df['session_intensity_index'] = (
        weights.w_explosiveness * df['explosiveness_z']
        + weights.w_repeatability * df['repeatability_z']
        + weights.w_volume * df['volume_z']
    )
    
    return df


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def build_session_intensity_df(
    df_raw: pd.DataFrame,
    window_seconds_list: tuple = (10, 20, 30),
    weights: Optional[IntensityWeights] = None
) -> pd.DataFrame:
    """
    Complete pipeline: clean, compute MP, extract session metrics, compute intensity index.
    
    Args:
        df_raw: Raw long-form tracking DataFrame
        window_seconds_list: Window durations (seconds) for MDP computation
        weights: IntensityWeights for intensity index calculation
    
    Returns:
        Session-level DataFrame with intensity metrics
    """
    df = clean_base_df(df_raw)
    df = compute_mp_from_equation(df)
    df = add_session_id(df)
    
    session_df = build_session_summary_df(df, mp_col='mp_eq', window_seconds_list=window_seconds_list)
    session_df = add_intensity_components(session_df, weights=weights)
    
    return session_df


# ============================================================================
# OPTIONAL MAIN CHECK
# ============================================================================

if __name__ == '__main__':
    import os
    
    csv_path = 'full_players_df.csv'
    if os.path.exists(csv_path):
        print(f"Loading {csv_path}...")
        df_raw = pd.read_csv(csv_path)
        
        print("Running pipeline...")
        intensity_df = build_session_intensity_df(df_raw)
        
        print("\nSession Intensity Index Results:")
        print(intensity_df.head())
        print(f"\nColumns: {list(intensity_df.columns)}")
        print(f"\nShape: {intensity_df.shape}")
    else:
        print(f"File {csv_path} not found. Run from the Grow Irish directory.")
