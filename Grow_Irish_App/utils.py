"""
Shared utilities for Grow Irish Session Intensity & MDP Explorer

Contains data loading, processing, filtering, and visualization functions
used across all pages of the multi-page Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple

from mp_intensity_pipeline import build_session_intensity_df, IntensityWeights
from src.config import WINDOW_COLOR_MAP, get_window_color


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_clean(files: List) -> pd.DataFrame:
    """
    Load and clean raw tracking data from CSV files.
    
    Performs:
    - Concatenates multiple CSVs
    - Splits 'file' column into date, player_id, event
    - Removes prefixes/suffixes from player_id and event
    - Converts date to datetime
    - Creates player_number mapping
    - Drops altitude column if present
    """
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    if 'file' in full_df.columns:
        full_df[['date', 'player_id', 'event']] = full_df['file'].str.split(r"/", expand=True)
        full_df['player_id'] = full_df['player_id'].str.replace(r'player_', "", regex=True)
        full_df['event'] = full_df['event'].str.replace(r'.json.gz', "", regex=True)
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df['player_number'] = pd.factorize(full_df['player_id'])[0] + 1
    
    if 'altitude' in full_df.columns:
        full_df = full_df.drop(['altitude'], axis=True)
    
    return full_df


@st.cache_data(show_spinner=False)
def load_default_data(path: str = "full_players_df.csv") -> pd.DataFrame:
    """Load default session-level intensity data from CSV."""
    df = pd.read_csv(path)
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(show_spinner=False)
def get_session_intensity_df(
    raw_path: str,
    w_explosiveness: float = 0.30,
    w_repeatability: float = 0.50,
    w_volume: float = 0.20
) -> pd.DataFrame:
    """
    Load raw data and compute session intensity metrics.
    Uses caching to avoid recomputation.
    """
    if isinstance(raw_path, str):
        raw = pd.read_csv(raw_path)
    else:
        raw = raw_path  # Already a DataFrame
    
    weights = IntensityWeights(
        w_explosiveness=w_explosiveness,
        w_repeatability=w_repeatability,
        w_volume=w_volume
    )
    return build_session_intensity_df(raw, weights=weights)


@st.cache_data(show_spinner=False)
def calculate_intensity_and_windows(
    _full_df: pd.DataFrame,
    w_speed: float,
    w_acc: float,
    w_hr: float,
    w_mp: float,
    columns_to_plot: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate intensity scores with rolling windows - cached for efficiency.
    
    Returns:
        - hr_df: DataFrame with intensity and rolling window columns
        - scaled_df: Scaled values DataFrame
    """
    scaler = MinMaxScaler()
    hr_df = _full_df.dropna(subset=['hr']).copy()
    
    # Scale values
    scaled_values = scaler.fit_transform(hr_df[columns_to_plot])
    scaled_df = pd.DataFrame(scaled_values, columns=columns_to_plot, index=hr_df.index)
    
    # Calculate intensity
    scaled_df['intensity'] = (
        w_speed * scaled_df['speed'] +
        w_acc * scaled_df['acc'] +
        w_hr * scaled_df['hr'] +
        w_mp * scaled_df['mp']
    )
    
    # Add intensity to original df
    hr_df['intensity'] = scaled_df['intensity'].values
    
    # Rolling windows
    if 'timestamp' in hr_df.columns:
        hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'], unit='ms', errors='coerce')
        hr_df = hr_df.sort_values(['player_id', 'timestamp']).reset_index(drop=True)
        
        # Calculate all rolling windows at once
        for window_size in [5, 10, 20, 30]:
            roll = (
                hr_df.groupby('player_id')
                .rolling(f'{window_size}s', on='timestamp')['intensity']
                .mean()
            )
            hr_df[f'intensity_{window_size}s'] = roll.to_numpy()
    
    return hr_df, scaled_df


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_filters(
    df: pd.DataFrame,
    selected_players: list,
    date_range: tuple,
    high_intensity_only: bool = False
) -> pd.DataFrame:
    """
    Apply player, date, and intensity filters to session intensity DataFrame.
    """
    filtered = df.copy()
    
    # Player filter
    if selected_players:
        filtered = filtered[filtered['player_id'].isin(selected_players)]
    
    # Date filter
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['date'] >= pd.Timestamp(start_date)) &
            (filtered['date'] <= pd.Timestamp(end_date))
        ]
    
    # High intensity filter
    if high_intensity_only and len(filtered) > 0:
        threshold = filtered['session_intensity_index'].quantile(0.75)
        filtered = filtered[filtered['session_intensity_index'] >= threshold]
    
    return filtered.sort_values('session_intensity_index', ascending=False)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_intensity_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Plot session intensity index over time.
    If multiple players, show separate lines per player using coach-friendly display names.
    Assumes df has player_display and event_display columns.
    """
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data to display")
    
    fig = px.line(
        df.sort_values('date'),
        x='date',
        y='session_intensity_index',
        color='player_display',
        hover_name='event_display',
        hover_data={'player_display': True, 'date': True, 'session_intensity_index': ':.2f'},
        markers=True,
        title="Session Intensity Over Time",
        labels={'session_intensity_index': 'Intensity (z)', 'date': 'Date', 'player_display': 'Player'}
    )
    
    fig.update_layout(hovermode='x unified', height=500, legend_title_text='Player')
    return fig


def plot_mdp_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Plot MDP 10/20/30 second windows as grouped bars.
    Aggregates across all rows in dataframe.
    """
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data to display")
    
    mdp_data = pd.DataFrame({
        'MDP 10s': [df['mdp_10'].mean()],
        'MDP 20s': [df['mdp_20'].mean()],
        'MDP 30s': [df['mdp_30'].mean()]
    })
    
    fig = px.bar(
        mdp_data,
        y=['MDP 10s', 'MDP 20s', 'MDP 30s'],
        title="Average MDP Windows",
        labels={'value': 'Power (W)', 'variable': 'Window'},
        barmode='group'
    )
    
    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_intensity_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plot histogram of session intensity scores.
    """
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data to display")
    
    fig = px.histogram(
        df,
        x='session_intensity_index',
        nbins=20,
        title="Intensity Index Distribution",
        labels={'session_intensity_index': 'Intensity (z)'},
        opacity=0.7
    )
    
    fig.update_layout(height=400)
    return fig


def plot_player_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Plot load vs intensity scatter plot.
    Bubble size represents MDP 10s, color represents player using coach-friendly display names.
    Assumes df has player_display and event_display columns.
    """
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data to display")

    df = df.copy()

    # Clean up rows where size would be invalid (NaN)
    if "mdp_10" in df.columns:
        df = df[df["mdp_10"].notna()]

    # If everything got filtered out, show a friendly empty figure
    if len(df) == 0:
        return go.Figure().add_annotation(
            text="No sessions with valid MDP values to display"
        )

    fig = px.scatter(
        df,
        x="total_mp_load",
        y="session_intensity_index",
        size="mdp_10",
        color="player_display",
        hover_name="event_display",
        hover_data={
            "player_display": True,
            "date": True,
            "total_mp_load": ":.0f",
            "session_intensity_index": ":.2f",
            "mdp_10": ":.0f",
        },
        title="Session Intensity vs Total MP Load",
        labels={
            "total_mp_load": "Total MP Load (A.U.)",
            "session_intensity_index": "Intensity (z)",
            "mdp_10": "Peak 10s (W)",
            "player_display": "Player",
        },
    )

    fig.update_layout(height=500, legend_title_text="Player")
    return fig


def plot_rolling_window_lines(
    df: pd.DataFrame,
    window_options: List[str],
    mdp_info: Optional[List[dict]] = None,
    show_mdp_overlays: bool = True,
    active_mdp_windows: Optional[List[str]] = None
) -> go.Figure:
    """
    Plot rolling window intensity lines over time for a single player/session.
    
    Each window is assigned a consistent, high-contrast color from WINDOW_COLOR_MAP.
    This ensures 5s is always red, 10s always blue, etc., regardless of selection.
    
    Optionally adds color-coded MDP overlay regions.
    """
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data to display")
    
    fig = go.Figure()
    
    for window in window_options:
        if window in df.columns:
            # Format trace name (e.g. "intensity_10s" -> "10s window")
            trace_name = window.replace('intensity_', '').replace('s', 's window')
            
            # Get color from map (supports multiple naming conventions)
            color = get_window_color(trace_name) or get_window_color(window) or "#333333"
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[window],
                mode='lines',
                name=trace_name,
                opacity=0.7 if window == 'intensity' else 1.0,
                line=dict(color=color, width=2.0)  # Thicker lines for clarity
            ))
    
    fig.update_layout(
        title="Intensity Over Time",
        xaxis_title="Time",
        yaxis_title="Intensity",
        hovermode='x unified',
        height=600,
        legend=dict(
            title_text="Intensity window",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=0.99,
        )
    )
    
    # Add MDP overlays if provided
    if mdp_info and show_mdp_overlays:
        fig = add_mdp_overlays_to_plot(
            fig,
            mdp_info,
            show_overlays=True,
            active_windows=active_mdp_windows
        )
    
    return fig


# ============================================================================
# METRICS & STATISTICS
# ============================================================================

def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """
    Compute summary metrics for current filtered view.
    """
    if len(df) == 0:
        return {
            'n_sessions': 0,
            'avg_intensity': 0.0,
            'max_intensity': 0.0,
            'avg_mdp_10': 0.0
        }
    
    return {
        'n_sessions': len(df),
        'avg_intensity': df['session_intensity_index'].mean(),
        'max_intensity': df['session_intensity_index'].max(),
        'avg_mdp_10': df['mdp_10'].mean()
    }


# ============================================================================
# WITHIN-SESSION MDP & TREND ANALYSIS (PHASE 2)
# ============================================================================

def compute_session_mdp_info(df_session: pd.DataFrame) -> List[dict]:
    """
    Compute MDP (Most Demanding Period) for each intensity window in a session.
    
    For each window (5s, 10s, 20s, 30s), finds the timestamp of maximum value
    and formats the time offset as mm:ss.
    
    Args:
        df_session: DataFrame for single player + session, sorted by timestamp
    
    Returns:
        List of dicts with keys: window, value, timestamp, time_offset_str,
        session_duration_s, pct_elapsed, has_warning
    """
    if df_session.empty:
        return []
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_session['timestamp']):
        df_session = df_session.copy()
        df_session['timestamp'] = pd.to_datetime(df_session['timestamp'], errors='coerce')
    
    session_start = df_session['timestamp'].min()
    session_end = df_session['timestamp'].max()
    session_duration = (session_end - session_start).total_seconds()
    
    mdp_info = []
    
    for window_size in [5, 10, 20, 30]:
        col_name = f'intensity_{window_size}s'
        
        # Skip if column doesn't exist or all NaN
        if col_name not in df_session.columns or df_session[col_name].isna().all():
            continue
        
        # Find max value and its timestamp
        max_idx = df_session[col_name].idxmax()
        max_val = df_session.loc[max_idx, col_name]
        max_time = df_session.loc[max_idx, 'timestamp']
        
        # Compute offset from session start
        time_offset = (max_time - session_start).total_seconds()
        minutes = int(time_offset // 60)
        seconds = int(time_offset % 60)
        time_offset_str = f"{minutes:02d}:{seconds:02d}"
        
        # Compute percentage elapsed
        pct_elapsed = (time_offset / session_duration * 100) if session_duration > 0 else 0
        
        # Anomaly flag: peak very early or very late
        has_warning = pct_elapsed < 10 or pct_elapsed > 90
        
        mdp_info.append({
            'window': f'{window_size}s',
            'value': float(max_val),  # type: ignore
            'timestamp': max_time,
            'time_offset_str': time_offset_str,
            'session_duration_s': session_duration,
            'pct_elapsed': pct_elapsed,
            'has_warning': has_warning
        })
    
    return mdp_info


def compute_early_late_comparison(
    df_session: pd.DataFrame,
    primary_window: str = "intensity_20s",
    segment_threshold: float = 0.33
) -> dict:
    """
    Compare early (first 33%) vs late (last 33%) intensity in a session.
    
    Args:
        df_session: DataFrame for single player + session
        primary_window: Column to analyze (e.g., "intensity_20s")
        segment_threshold: Fraction for early/late split (default 0.33 = thirds)
    
    Returns:
        Dict with early_mean, late_mean, mid_mean, delta_pct, sample counts,
        warning flags, and linguistic interpretation
    """
    if df_session.empty or primary_window not in df_session.columns:
        return {
            'early_mean': np.nan,
            'mid_mean': np.nan,
            'late_mean': np.nan,
            'delta_pct': np.nan,
            'early_samples': 0,
            'mid_samples': 0,
            'late_samples': 0,
            'early_warning': True,
            'late_warning': True,
            'interpretation': 'Insufficient data'
        }
    
    n = len(df_session)
    threshold_idx = int(n * segment_threshold)
    
    # Split into early, mid, late thirds
    early = df_session.iloc[:threshold_idx]
    mid = df_session.iloc[threshold_idx:n-threshold_idx]
    late = df_session.iloc[n-threshold_idx:]
    
    # Compute means, excluding NaN
    early_mean = early[primary_window].mean()
    mid_mean = mid[primary_window].mean()
    late_mean = late[primary_window].mean()
    
    # Compute trend
    if pd.notna(early_mean) and pd.notna(late_mean) and early_mean > 1e-6:
        delta_pct = (late_mean - early_mean) / early_mean * 100
    else:
        delta_pct = 0.0
    
    # Sample count warnings (flag if < 20 samples)
    early_warning = len(early) < 20
    late_warning = len(late) < 20
    
    # Linguistic interpretation
    if delta_pct < -15:
        interpretation = "Fading 😴"
    elif delta_pct > 15:
        interpretation = "Strong finish 💪"
    else:
        interpretation = "Consistent 📊"
    
    return {
        'early_mean': early_mean,
        'mid_mean': mid_mean,
        'late_mean': late_mean,
        'delta_pct': delta_pct,
        'early_samples': len(early),
        'mid_samples': len(mid),
        'late_samples': len(late),
        'early_warning': early_warning,
        'late_warning': late_warning,
        'interpretation': interpretation
    }


def add_mdp_overlays_to_plot(
    fig: go.Figure,
    mdp_info: List[dict],
    show_overlays: bool = True,
    active_windows: Optional[List[str]] = None
) -> go.Figure:
    """
    Add color-coded vrect shaded regions for each MDP window to Plotly figure.
    
    Colors: 5s=lightcyan, 10s=lightblue, 20s=cornflowerblue, 30s=darkblue
    Each region spans ±(window_duration/2) around peak timestamp, clipped to
    session bounds.
    
    Args:
        fig: Plotly Figure (go.Figure)
        mdp_info: List of dicts from compute_session_mdp_info()
        show_overlays: Whether to add overlays (default True)
        active_windows: List of windows to show (e.g., ['10s', '20s']). 
                       If None, show all available.
    
    Returns:
        Modified Plotly figure with vrect overlays
    """
    if not show_overlays or not mdp_info:
        return fig
    
    # Color map: gradient from light to dark blue
    color_map = {
        '5s': 'rgba(176, 224, 230, 0.15)',      # lightcyan
        '10s': 'rgba(173, 216, 230, 0.15)',     # lightblue
        '20s': 'rgba(100, 149, 237, 0.15)',     # cornflowerblue
        '30s': 'rgba(25, 25, 112, 0.15)',       # midnightblue (darkish)
    }
    
    # Filter by active windows if specified
    display_mdp = mdp_info
    if active_windows:
        display_mdp = [m for m in mdp_info if m['window'] in active_windows]
    
    # Get session bounds for clipping
    session_start = min([m['timestamp'] for m in mdp_info])
    session_end = max([m['timestamp'] for m in mdp_info])
    
    # Add vrect for each MDP
    for mdp in display_mdp:
        window = mdp['window']
        timestamp = mdp['timestamp']
        
        # Window duration in seconds
        window_duration = int(window.rstrip('s'))
        half_window = window_duration / 2.0
        
        # Compute region bounds
        from datetime import timedelta
        x0 = timestamp - timedelta(seconds=half_window)
        x1 = timestamp + timedelta(seconds=half_window)
        
        # Clip to session bounds
        x0 = max(x0, session_start)
        x1 = min(x1, session_end)
        
        # Add shaded region
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color_map.get(window, 'rgba(200, 200, 200, 0.15)'),
            layer='below',
            line_width=0,
            annotation_text=f"MDP {window}",
            annotation_position="top left",
            annotation_font_size=10,
        )
    
    return fig


def get_early_late_interpretation(early_mid_late_values: tuple) -> str:
    """
    Return linguistic interpretation of early/mid/late trend.
    
    Args:
        early_mid_late_values: Tuple of (early_mean, mid_mean, late_mean)
    
    Returns:
        Categorical interpretation: "Fading", "Consistent", "Strong finish", etc.
    """
    early, mid, late = early_mid_late_values
    
    if pd.isna(early) or pd.isna(late):
        return "Insufficient data"
    
    # Simple heuristic: compare late vs early
    if late < early * 0.85:  # Faded > 15%
        return "Fading 😴"
    elif late > early * 1.15:  # Stronger > 15%
        return "Strong finish 💪"
    else:
        return "Consistent 📊"
