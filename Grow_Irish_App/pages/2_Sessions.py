"""
Sessions Page - Interactive Session Explorer

Browse all sessions, apply filters, and visualize intensity trends.
Supports two view modes: Coach (focused story) and Analyst (full metrics).
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from collections import Counter
from utils import (
    get_session_intensity_df,
    apply_filters,
    plot_intensity_over_time,
    plot_intensity_distribution,
    plot_player_scatter
)
from src.display_names import build_player_display_map, add_player_display_column, add_event_display_column
from src.ui.nav import render_global_nav

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    st.session_state['raw_df'] = None
    st.session_state['session_df'] = None

if 'w_explosiveness' not in st.session_state:
    st.session_state['w_explosiveness'] = 0.30
    st.session_state['w_repeatability'] = 0.50
    st.session_state['w_volume'] = 0.20

# Initialize view mode if not already done
if "sessions_view_mode" not in st.session_state:
    st.session_state["sessions_view_mode"] = "Coach"

st.set_page_config(page_title="Sessions", layout="wide")
render_global_nav(current_page="sessions")
st.title("📊 Sessions Explorer")

# ============================================================================
# GLOBAL VIEW MODE TOGGLE (Coach vs. Analyst)
# ============================================================================

view_mode = st.radio(
    "View mode",
    ["Coach", "Analyst"],
    horizontal=True,
    key="sessions_view_mode",
    help="Coach = quick story and decisions. Analyst = full metrics, charts, and exports."
)

if view_mode == "Coach":
    st.caption("🏃 Coach mode: quick overview and key decisions. Switch to Analyst mode for full metrics and exports.")
else:
    st.caption("🔬 Analyst mode: full metrics, charts, and export tools for deeper analysis.")

st.markdown("---")

# ============================================================================
# HELPER FUNCTIONS (Modularized for clean rendering)
# ============================================================================

def render_filters_and_presets(session_df: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    """
    Render filter bar, apply filters, and return filtered dataframe.
    """
    st.markdown("### 🔍 Filters")

    # Reset filters button (must come BEFORE widgets to avoid session_state conflicts)
    fcb1, fcb2 = st.columns([1, 4])
    with fcb1:
        if st.button("Reset filters", key="reset_filters_button"):
            st.session_state['sessions_dates'] = None
            st.session_state['sessions_players'] = []
            st.session_state['sessions_high_intensity'] = False
            st.rerun()

    fc1, fc2, fc3 = st.columns(3)

    # Compute date bounds first
    if len(session_df) > 0 and 'date' in session_df.columns:
        min_date = session_df['date'].min().date()
        max_date = session_df['date'].max().date()
        
        # Preset buttons in their own row
        pf1, pf2, pf3, pf4 = st.columns([1, 1, 1, 1])
        
        with pf1:
            if st.button("Last 7 days", key="sessions_preset_7d"):
                st.session_state['sessions_dates'] = (max_date - datetime.timedelta(days=6), max_date)
                st.rerun()
        
        with pf2:
            if st.button("Last 14 days", key="sessions_preset_14d"):
                st.session_state['sessions_dates'] = (max_date - datetime.timedelta(days=13), max_date)
                st.rerun()
        
        with pf3:
            if st.button("Last 30 days", key="sessions_preset_30d"):
                st.session_state['sessions_dates'] = (max_date - datetime.timedelta(days=29), max_date)
                st.rerun()
        
        with pf4:
            if st.button("Full range", key="sessions_preset_full"):
                st.session_state['sessions_dates'] = (min_date, max_date)
                st.rerun()
        
        st.caption("Use presets to jump between recent training windows.")

    # Date range filter
    with fc1:
        if len(session_df) > 0:
            min_date = session_df['date'].min().date()
            max_date = session_df['date'].max().date()
            
            # Get stored value or use defaults
            stored_dates = st.session_state.get('sessions_dates', None)
            
            # Validate stored dates are within current bounds
            if stored_dates is not None:
                try:
                    if (len(stored_dates) == 2 and 
                        stored_dates[0] >= min_date and 
                        stored_dates[1] <= max_date):
                        default_dates = stored_dates
                    else:
                        default_dates = (min_date, max_date)
                except (TypeError, AttributeError):
                    default_dates = (min_date, max_date)
            else:
                default_dates = (min_date, max_date)
            
            date_range = st.date_input(
                "Date range",
                value=default_dates,
                min_value=min_date,
                max_value=max_date,
                key="sessions_dates"
            )
        else:
            date_range = None

    # Player filter (using display names for coach-friendly UI)
    with fc2:
        all_players = sorted(session_df['player_id'].unique().tolist())
        player_display_map = build_player_display_map(session_df, player_col='player_id')
        
        # Build reverse mapping: display label -> internal id
        player_options = {label: pid for pid, label in player_display_map.items()}
        
        stored_players = st.session_state.get('sessions_players', None)
        if stored_players is None or len(stored_players) == 0:
            default_labels = list(player_options.keys())
        else:
            # Convert stored player_ids back to labels
            default_labels = [player_display_map.get(pid, f"Unknown {pid}") for pid in stored_players]
        
        selected_labels = st.multiselect(
            "Player(s)",
            options=list(player_options.keys()),
            default=default_labels,
            key="sessions_players_display"
        )
        
        # Convert back to internal player_ids for downstream logic
        selected_players = [player_options[label] for label in selected_labels]

    # Optional: High-intensity filter
    with fc3:
        high_intensity_only = st.checkbox(
            "Show only highest intensity sessions (top 25%)",
            value=False,
            key="sessions_high_intensity",
            help="When checked, only sessions in roughly the top quartile of the intensity index are shown."
        )

    st.caption("All metrics below respect the filters selected here.")

    # Apply filters to create filtered_df
    filtered_df = apply_filters(
        session_df,
        selected_players,
        date_range,
        high_intensity_only
    )
    
    return filtered_df


def render_summary_metrics_and_narrative(filtered_df: pd.DataFrame) -> None:
    """
    Render summary metrics and training block narrative (shown in both modes).
    """
    st.markdown("## 📈 Summary Metrics")
    
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Sessions in view", len(filtered_df))

    with c2:
        st.metric(
            "Avg session intensity (z)",
            f"{filtered_df['session_intensity_index'].mean():.2f}",
            help="Z-score of overall session demand; 0 is typical, > 1 is hard, < -1 is light."
        )

    with c3:
        st.metric(
            "Most intense session (z)",
            f"{filtered_df['session_intensity_index'].max():.2f}",
            help="Highest session intensity z-score in this filtered view."
        )

    with c4:
        st.metric(
            "Avg peak 10s power (W)",
            f"{filtered_df['mdp_10'].mean():.0f}",
            help="Average of peak 10-second power (MDP 10s) across the sessions in view."
        )

    st.caption(
        "Session intensity is a z-scored index of overall demand. "
        "MDP peaks are in watts; higher values mean greater power demands."
    )

    # Training block narrative + distribution
    if not filtered_df.empty and 'session_intensity_index' in filtered_df.columns:
        n_sessions = len(filtered_df)
        avg_intensity = filtered_df['session_intensity_index'].mean()
        max_intensity = filtered_df['session_intensity_index'].max()
        
        # Count sessions by intensity band
        very_hard = (filtered_df['session_intensity_index'] > 1.5).sum()
        hard = ((filtered_df['session_intensity_index'] > 1.0) & (filtered_df['session_intensity_index'] <= 1.5)).sum()
        light = (filtered_df['session_intensity_index'] < -0.5).sum()
        
        st.markdown(
            f"Across **{n_sessions}** sessions in this view, average session intensity was "
            f"**{avg_intensity:.2f} z**. This block included **{hard}** hard sessions (z > 1.0), "
            f"**{very_hard}** very hard sessions (z > 1.5), and **{light}** light sessions (z < -0.5). "
            f"The hardest session reached **{max_intensity:.2f} z**."
        )
        
        # Distribution micro-panel
        dbc1, dbc2, dbc3 = st.columns(3)
        with dbc1:
            st.metric("🔴 Very hard (z > 1.5)", very_hard)
        with dbc2:
            st.metric("🟡 Hard (1.0 < z ≤ 1.5)", hard)
        with dbc3:
            st.metric("🟢 Light (z < -0.5)", light)
        
        st.caption("Session counts by intensity band for the current filters.")


def render_sessions_summary_table(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render sessions summary table with mode-aware column selection.
    Coach mode shows trimmed columns; Analyst mode shows full detail.
    Uses display names for players and events (coach-friendly labels).
    """
    st.markdown("## 📋 Sessions Summary")

    if filtered_df.empty:
        st.info("No sessions to display.")
        return

    # Build display name mappings
    player_display_map = build_player_display_map(filtered_df, player_col='player_id')
    display_df = add_player_display_column(filtered_df, player_display_map)
    display_df = add_event_display_column(display_df)

    base_cols = [
        'player_display', 'date', 'event_display',
        'session_intensity_index', 'mdp_10', 'total_mp_load'
    ]

    display_df = display_df[base_cols].copy()

    # Add Session tag column
    def classify_session(row):
        z = row.get('session_intensity_index', None)
        if z is None or pd.isna(z):
            return "Unclassified"
        if z > 1.5:
            return "Very hard"
        if z > 1.0:
            return "Hard"
        if z < -0.5:
            return "Light"
        return "Typical"

    display_df['Session tag'] = display_df.apply(classify_session, axis=1)

    # Rename for coach-friendly display
    display_df = display_df.rename(columns={
        'player_display': 'Player',
        'date': 'Date',
        'event_display': 'Session',
        'session_intensity_index': 'Session intensity (z)',
        'mdp_10': 'Peak 10s (W)',
        'total_mp_load': 'Total load (A.U.)'
    })

    # Reorder columns
    col_order = [
        'Player', 'Date', 'Session',
        'Session intensity (z)', 'Session tag',
        'Peak 10s (W)', 'Total load (A.U.)'
    ]
    display_df = display_df[col_order]

    # Format numeric columns
    display_df = display_df.round({
        'Session intensity (z)': 2,
        'Peak 10s (W)': 0,
        'Total load (A.U.)': 0
    })

    # Sort by intensity descending
    display_df = display_df.sort_values('Session intensity (z)', ascending=False)

    # Mode-aware column selection for on-screen display
    if view_mode == "Coach":
        coach_cols = [
            'Player', 'Date', 'Session',
            'Session intensity (z)', 'Session tag',
            'Peak 10s (W)', 'Total load (A.U.)'
        ]
        coach_cols = [c for c in coach_cols if c in display_df.columns]
        display_table = display_df[coach_cols]
    else:
        display_table = display_df

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.caption("Table sorted by Session intensity (z), highest first. Session tag is based on session intensity z-score thresholds.")

    # CSV Export (always available)
    csv_export = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download all metrics (CSV)",
        data=csv_export,
        file_name="sessions_export.csv",
        mime="text/csv"
    )


def render_advanced_metrics_section(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render advanced metrics expander (Analyst mode only).
    Displays compact table on-screen; exports full dataframe with all columns.
    """
    if view_mode != "Analyst":
        return

    with st.expander("Session type by player (compact view)"):
        required_advanced_cols = [
            'player_id', 'session_id', 'date',
            'session_intensity_index', 'explosiveness_z', 'volume_z'
        ]
        
        has_peak_data = 'mdp_peak_value' in filtered_df.columns or 'mdp_20' in filtered_df.columns
        has_repeatability = 'repeatability_z' in filtered_df.columns
        
        missing_cols = [col for col in required_advanced_cols if col not in filtered_df.columns]
        
        if filtered_df.empty or missing_cols or not has_peak_data:
            st.info("Advanced session type metrics are unavailable for the current filters.")
        
        else:
            # Build full advanced dataframe (for CSV export)
            full_cols = [
                'player_id', 'session_id', 'date',
                'mdp_10', 'mdp_20', 'mdp_30'
            ]
            
            if 'mdp_peak_value' in filtered_df.columns:
                full_cols.append('mdp_peak_value')
            if 'mdp_peak_window' in filtered_df.columns:
                full_cols.append('mdp_peak_window')
            
            full_cols.extend([
                'explosiveness_z', 'repeatability_z', 'volume_z',
                'session_intensity_index'
            ])
            
            advanced_full_df = filtered_df[full_cols].copy()
            
            full_rename_dict = {
                'player_id': 'Player',
                'session_id': 'Session',
                'mdp_10': 'MDP 10s (W)',
                'mdp_20': 'MDP 20s (W)',
                'mdp_30': 'MDP 30s (W)',
                'explosiveness_z': 'Explosiveness (z)',
                'repeatability_z': 'Repeatability (z)',
                'volume_z': 'Volume (z)',
                'session_intensity_index': 'Session intensity (z)'
            }
            
            if 'mdp_peak_value' in advanced_full_df.columns:
                full_rename_dict['mdp_peak_value'] = 'MDP peak (W)'
            if 'mdp_peak_window' in advanced_full_df.columns:
                full_rename_dict['mdp_peak_window'] = 'MDP window'
            
            advanced_full_df = advanced_full_df.rename(columns=full_rename_dict)
            
            numeric_cols_full = [col for col in advanced_full_df.columns if advanced_full_df[col].dtype in ['float64', 'int64']]
            advanced_full_df = advanced_full_df.round({col: 2 for col in numeric_cols_full})
            
            # Build compact advanced dataframe (for on-screen display)
            peak_col = 'mdp_peak_value' if 'mdp_peak_value' in filtered_df.columns else 'mdp_20'
            peak_label = 'Peak power (W)' if peak_col == 'mdp_peak_value' else 'Peak 20s (W)'
            
            compact_cols = [
                'player_display', 'event_display', 'date',
                peak_col, 'explosiveness_z', 'volume_z',
                'session_intensity_index'
            ]
            
            if has_repeatability:
                compact_cols.insert(5, 'repeatability_z')
            
            advanced_compact_df = filtered_df[compact_cols].copy()
            
            compact_rename_dict = {
                'player_display': 'Player',
                'event_display': 'Session',
                'date': 'Date',
                peak_col: peak_label,
                'explosiveness_z': 'Explosiveness (z)',
                'repeatability_z': 'Repeatability (z)',
                'volume_z': 'Volume (z)',
                'session_intensity_index': 'Session intensity (z)'
            }
            
            advanced_compact_df = advanced_compact_df.rename(columns=compact_rename_dict)
            
            z_cols = ['Explosiveness (z)', 'Volume (z)', 'Session intensity (z)']
            if has_repeatability:
                z_cols.insert(1, 'Repeatability (z)')
            
            advanced_compact_df = advanced_compact_df.dropna(subset=z_cols, how='all')
            
            numeric_cols_compact = [col for col in advanced_compact_df.columns if advanced_compact_df[col].dtype in ['float64', 'int64']]
            advanced_compact_df = advanced_compact_df.round({col: 2 for col in numeric_cols_compact})
            
            sort_cols = ['Session intensity (z)', 'Volume (z)', 'Player']
            sort_asc = [False, False, True]
            advanced_compact_df = advanced_compact_df.sort_values(by=sort_cols, ascending=sort_asc)
            
            st.dataframe(advanced_compact_df, use_container_width=True, hide_index=True)
            
            st.caption(
                "Each row shows how this session felt for that player: "
                "peak power in our primary window plus explosiveness, volume, and overall session intensity z-scores."
            )
            
            st.markdown("#### Export full advanced metrics")
            st.caption(
                "Download the full MDP 10s/20s/30s and component z-scores for deeper analysis in Python or Excel."
            )
            
            csv_full = advanced_full_df.to_csv(index=False)
            st.download_button(
                label="📥 Download full advanced metrics (CSV)",
                data=csv_full,
                file_name="advanced_metrics_full.csv",
                mime="text/csv",
                key="download_advanced_metrics_full"
            )


def render_intensity_over_time_section(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render intensity over time chart.
    Coach mode: team average only. Analyst mode: toggle between team average and all players.
    """
    if filtered_df.empty or 'session_intensity_index' not in filtered_df.columns:
        st.info("No intensity data available for the selected filters.")
        return

    st.markdown("## 📊 Visualizations")
    st.markdown("### Intensity Over Time")

    # Determine if we can show the toggle
    allow_all_players = (view_mode == "Analyst")
    unique_dates = filtered_df['date'].nunique()

    if allow_all_players and unique_dates >= 2:
        chart_view = st.radio(
            "View mode:",
            ["Team average", "All players"],
            horizontal=True,
            key="sessions_intensity_view_mode_chart"
        )
    else:
        chart_view = "Team average"

    if chart_view == "All players" and allow_all_players:
        # Per-player lines
        fig = plot_intensity_over_time(filtered_df)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Typical")
        fig.add_hline(y=1, line_dash="dot", line_color="red", annotation_text="Hard")
        fig.add_hline(y=-1, line_dash="dot", line_color="blue", annotation_text="Light")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Team average with ±1σ band
        if not filtered_df.empty and 'session_intensity_index' in filtered_df.columns and 'date' in filtered_df.columns:
            team_stats = filtered_df.groupby('date').agg({
                'session_intensity_index': ['mean', 'std']
            }).reset_index()
            team_stats.columns = ['date', 'mean_intensity', 'std_intensity']
            team_stats = team_stats.sort_values('date')
            
            team_stats['std_intensity'] = team_stats['std_intensity'].fillna(0)
            
            team_stats['upper_band'] = team_stats['mean_intensity'] + team_stats['std_intensity']
            team_stats['lower_band'] = team_stats['mean_intensity'] - team_stats['std_intensity']
            
            fig_team = go.Figure()
            
            fig_team.add_trace(go.Scatter(
                x=team_stats['date'],
                y=team_stats['upper_band'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig_team.add_trace(go.Scatter(
                x=team_stats['date'],
                y=team_stats['lower_band'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor='rgba(68, 68, 68, 0.2)',
                name='±1σ band'
            ))
            
            fig_team.add_trace(go.Scatter(
                x=team_stats['date'],
                y=team_stats['mean_intensity'],
                mode='lines+markers',
                name='Team average',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            fig_team.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Typical")
            fig_team.add_hline(y=1, line_dash="dot", line_color="red", annotation_text="Hard")
            fig_team.add_hline(y=-1, line_dash="dot", line_color="blue", annotation_text="Light")
            
            fig_team.update_layout(
                xaxis_title="Date",
                yaxis_title="Session intensity (z)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_team, use_container_width=True)
        else:
            st.warning("Cannot compute team average without date and intensity data.")


def render_regression_and_interpretation(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render regression scatter plot and coaching interpretation (Analyst mode only).
    """
    if view_mode != "Analyst":
        return

    if filtered_df.empty:
        return

    st.markdown("### Session intensity vs total load")
    
    fig = plot_player_scatter(filtered_df)
    
    fig.update_layout(
        xaxis_title="Total load (A.U.)",
        yaxis_title="Session intensity (z)"
    )
    
    x = filtered_df["total_mp_load"].astype(float).values
    y = filtered_df["session_intensity_index"].astype(float).values
    
    valid_idx = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    
    if len(x_clean) > 1:
        b, a = np.polyfit(x_clean, y_clean, 1)
        y_pred = a + b * x_clean
        
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        xs = np.linspace(x_clean.min(), x_clean.max(), 100)
        ys = a + b * xs
        
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Best fit",
            line=dict(color="red", dash="dash", width=2)
        ))
        
        eq_str = f"Session intensity (z) = {a:.2f} + {b:.4f} × Total load"
        r2_str = f"R² = {r2:.3f}"
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"**Best-fit line:** {eq_str}  |  **{r2_str}**  "
            "Higher R² means total load explains more of the variation in session intensity."
        )
        
        trend_desc = "no clear relationship"
        if b > 0.0005:
            trend_desc = "higher total load is generally associated with higher session intensity"
        elif b < -0.0005:
            trend_desc = "higher total load is surprisingly associated with *lower* measured intensity"
        
        st.markdown(
            f"**Coaching interpretation:** In this filtered view, "
            f"{trend_desc}. The current R² of **{r2:.2f}** means that "
            f"total load explains about **{r2*100:.0f}%** of the variation in session intensity."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Not enough data to compute a reliable regression line for this view.")


def render_intensity_distribution_histogram(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render intensity distribution histogram (Analyst mode only).
    NOTE: This function has been deprecated and no longer renders any output.
    The distribution chart has been removed from the Analyst view.
    """
    pass


def render_player_load_overview(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render player load overview table (Analyst mode only).
    Uses player display names for coach-friendly labels.
    """
    if view_mode != "Analyst":
        return

    if not filtered_df.empty and 'player_id' in filtered_df.columns and 'session_intensity_index' in filtered_df.columns and 'total_mp_load' in filtered_df.columns:
        with st.expander("Player load overview (filtered range)"):
            player_summary = (
                filtered_df
                .groupby('player_id')
                .agg(
                    sessions=('session_id', 'nunique'),
                    avg_intensity=('session_intensity_index', 'mean'),
                    avg_load=('total_mp_load', 'mean')
                )
                .reset_index()
            )
            
            # Add display names for coach-friendly labels
            player_display_map = build_player_display_map(filtered_df, player_col='player_id')
            player_summary['player_display'] = player_summary['player_id'].map(player_display_map)
            
            player_summary = player_summary.rename(columns={
                'player_display': 'Player',
                'sessions': 'Sessions',
                'avg_intensity': 'Avg session intensity (z)',
                'avg_load': 'Avg total load (A.U.)'
            })
            
            # Keep only display columns (drop internal player_id)
            player_summary = player_summary[['Player', 'Sessions', 'Avg session intensity (z)', 'Avg total load (A.U.)']]
            
            player_summary = player_summary.sort_values(
                by=['Sessions', 'Avg session intensity (z)'],
                ascending=[False, False]
            )
            
            player_summary = player_summary.round({
                'Avg session intensity (z)': 2,
                'Avg total load (A.U.)': 0
            })
            
            st.dataframe(
                player_summary,
                use_container_width=True,
                hide_index=True
            )
            st.caption("How often and how hard each player has trained in the current filtered block.")


def render_performance_highlights_and_risk_radar(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render performance highlights and player risk radar (mode-aware detail level).
    Uses player_display and event_display for coach-friendly labels.
    Assumes filtered_df already has player_display and event_display columns.
    """
    if filtered_df.empty:
        return

    st.markdown("## ⚡ Performance Highlights")

    # Create four top-3 dataframes (using display names for coach-friendly labels)
    df_top_peak10s = filtered_df.nlargest(3, 'mdp_10')[[
        'player_display', 'date', 'event_display', 'mdp_10', 'session_intensity_index'
    ]].copy()
    df_top_peak10s = df_top_peak10s.rename(columns={
        'player_display': 'Player',
        'date': 'Date',
        'event_display': 'Session',
        'mdp_10': 'Peak 10s (W)',
        'session_intensity_index': 'Session intensity (z)'
    })
    df_top_peak10s = df_top_peak10s.round({'Peak 10s (W)': 0, 'Session intensity (z)': 2})

    headline_peak10s = None
    if not df_top_peak10s.empty:
        top_peak10_val = df_top_peak10s.iloc[0]['Peak 10s (W)']
        top_peak10_player = df_top_peak10s.iloc[0]['Player']
        top_peak10_session = df_top_peak10s.iloc[0]['Session']
        headline_peak10s = f"Highest peak 10s effort: {top_peak10_val:.0f} W ({top_peak10_player}, {top_peak10_session})"
        
    df_sustained_temp = filtered_df.copy()
    df_sustained_temp['sustained_avg'] = 0.5 * (df_sustained_temp['mdp_20'] + df_sustained_temp['mdp_30'])
    df_top_sustained = df_sustained_temp.nlargest(3, 'sustained_avg')[[
        'player_display', 'date', 'event_display', 'mdp_20', 'session_intensity_index'
    ]].copy()
    df_top_sustained = df_top_sustained.rename(columns={
        'player_display': 'Player',
        'date': 'Date',
        'event_display': 'Session',
        'mdp_20': 'Peak 20s (W)',
        'session_intensity_index': 'Session intensity (z)'
    })
    df_top_sustained = df_top_sustained.round({'Peak 20s (W)': 0, 'Session intensity (z)': 2})

    headline_sustained = None
    if not df_top_sustained.empty:
        top_sustained_val = df_top_sustained.iloc[0]['Peak 20s (W)']
        top_sustained_player = df_top_sustained.iloc[0]['Player']
        top_sustained_session = df_top_sustained.iloc[0]['Session']
        headline_sustained = f"Best sustained effort: {top_sustained_val:.0f} W ({top_sustained_player}, {top_sustained_session})"
        
    df_top_load = filtered_df.nlargest(3, 'total_mp_load')[[
        'player_display', 'date', 'event_display', 'total_mp_load', 'session_intensity_index'
    ]].copy()
    df_top_load = df_top_load.rename(columns={
        'player_display': 'Player',
        'date': 'Date',
        'event_display': 'Session',
        'total_mp_load': 'Total load (A.U.)',
        'session_intensity_index': 'Session intensity (z)'
    })
    df_top_load = df_top_load.round({'Total load (A.U.)': 0, 'Session intensity (z)': 2})

    headline_load = None
    if not df_top_load.empty:
        top_load_val = df_top_load.iloc[0]['Total load (A.U.)']
        top_load_player = df_top_load.iloc[0]['Player']
        top_load_session = df_top_load.iloc[0]['Session']
        headline_load = f"Biggest workload: {top_load_val:.0f} A.U. ({top_load_player}, {top_load_session})"

    df_top_mdp_peak = pd.DataFrame()
    headline_mdp_peak = None
    has_mdp_peak = False
    
    if 'mdp_peak_value' in filtered_df.columns:
        df_top_mdp_peak = filtered_df.nlargest(3, 'mdp_peak_value')[[
            'player_display', 'date', 'event_display', 'mdp_peak_value', 'session_intensity_index'
        ]].copy()
        df_top_mdp_peak = df_top_mdp_peak.rename(columns={
            'player_display': 'Player',
            'date': 'Date',
            'event_display': 'Session',
            'mdp_peak_value': 'MDP peak (W)',
            'session_intensity_index': 'Session intensity (z)'
        })
        df_top_mdp_peak = df_top_mdp_peak.round({'MDP peak (W)': 0, 'Session intensity (z)': 2})
        
        if not df_top_mdp_peak.empty:
            top_peak_val = df_top_mdp_peak.iloc[0]['MDP peak (W)']
            top_peak_player = df_top_mdp_peak.iloc[0]['Player']
            top_peak_session = df_top_mdp_peak.iloc[0]['Session']
            headline_mdp_peak = f"Absolute hardest session: {top_peak_val:.0f} W ({top_peak_player}, {top_peak_session})"
        
        has_mdp_peak = True

    # Layout: 2×2 grid in Analyst mode, headlines only in Coach mode
    if view_mode == "Analyst":
        # Full tables
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        with r1c1:
            st.markdown("#### Most Explosive Efforts (Top 3)")
            if headline_peak10s:
                st.caption(headline_peak10s)
            st.caption("Top 3 highest 10-second MDP peaks recorded.")
            st.dataframe(df_top_peak10s, use_container_width=True, hide_index=True)

        with r1c2:
            st.markdown("#### Best Sustained Effort (Top 3)")
            if headline_sustained:
                st.caption(headline_sustained)
            st.caption("Top 3 sessions based on 20–30 second MDP average.")
            st.dataframe(df_top_sustained, use_container_width=True, hide_index=True)

        with r2c1:
            st.markdown("#### Biggest Workloads (Top 3)")
            if headline_load:
                st.caption(headline_load)
            st.caption("Top 3 sessions with the highest total accumulated load.")
            st.dataframe(df_top_load, use_container_width=True, hide_index=True)

        with r2c2:
            if has_mdp_peak:
                st.markdown("#### Hardest Sessions Overall (Top 3)")
                if headline_mdp_peak:
                    st.caption(headline_mdp_peak)
                st.caption("Top 3 sessions by single highest MDP peak.")
                st.dataframe(df_top_mdp_peak, use_container_width=True, hide_index=True)
            else:
                st.markdown("#### Hardest Sessions Overall (Top 3)")
                st.caption("Data not available for this view.")
                st.info("⚠️ MDP peak data not computed for the current session set.")
    else:
        # Coach mode: show headlines only
        st.markdown("### Key Performances")
        if headline_peak10s:
            st.write(f"🔥 {headline_peak10s}")
        if headline_sustained:
            st.write(f"💪 {headline_sustained}")
        if headline_load:
            st.write(f"⚙️ {headline_load}")
        if headline_mdp_peak:
            st.write(f"⚡ {headline_mdp_peak}")

    # Player Risk Radar (both modes)
    if not filtered_df.empty:
        all_highlight_players = []
        
        if not df_top_peak10s.empty:
            all_highlight_players.extend(df_top_peak10s['Player'].tolist())
        if not df_top_sustained.empty:
            all_highlight_players.extend(df_top_sustained['Player'].tolist())
        if not df_top_load.empty:
            all_highlight_players.extend(df_top_load['Player'].tolist())
        if has_mdp_peak and not df_top_mdp_peak.empty:
            all_highlight_players.extend(df_top_mdp_peak['Player'].tolist())
        
        if all_highlight_players:
            player_counts = Counter(all_highlight_players)
            flagged_players = {p: count for p, count in player_counts.items() if count >= 2}
            
            if flagged_players:
                st.markdown("#### 🚨 Player Risk Radar")
                st.caption("Players appearing in multiple highlight categories may need monitoring.")
                
                for player_display_label in sorted(flagged_players.keys()):
                    count = flagged_players[player_display_label]
                    # Map display label back to internal player_id for data lookup
                    player_display_map_inv = {v: k for k, v in build_player_display_map(filtered_df).items()}
                    internal_player_id = player_display_map_inv.get(player_display_label)
                    
                    if internal_player_id is not None:
                        player_data = filtered_df[filtered_df['player_id'] == internal_player_id]
                        
                        if not player_data.empty and 'session_intensity_index' in player_data.columns:
                            player_mean_intensity = player_data['session_intensity_index'].mean()
                            st.write(
                                f"- **{player_display_label}** appears in **{count}** highlight categories "
                                f"with avg intensity **{player_mean_intensity:.2f} z**"
                            )
            else:
                st.caption("No players stand out across multiple highlight categories. Team distribution is balanced.")


def render_view_exports_section(filtered_df: pd.DataFrame, view_mode: str) -> None:
    """
    Render exports for this view (Analyst mode only).
    """
    if view_mode != "Analyst":
        return

    if not filtered_df.empty:
        with st.expander("📥 Exports for this view"):
            # A. Export sessions in this view (with display names)
            export_sessions = filtered_df.copy()
            export_sessions = export_sessions[[
                'player_display', 'date', 'event_display',
                'session_intensity_index', 'mdp_10', 'total_mp_load'
            ]].copy()
            
            export_sessions = export_sessions.rename(columns={
                'player_display': 'Player',
                'date': 'Date',
                'event_display': 'Session',
                'session_intensity_index': 'Session intensity (z)',
                'mdp_10': 'Peak 10s (W)',
                'total_mp_load': 'Total load (A.U.)'
            })
            
            csv_sessions = export_sessions.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered sessions (CSV)",
                data=csv_sessions,
                file_name="sessions_filtered_view.csv",
                mime="text/csv",
                key="download_sessions_filtered"
            )
            
            # B. Export highlight tables stitched together
            highlight_frames = []
            
            if 'df_top_peak10s' in globals() and not pd.DataFrame().empty:
                # Need to rebuild from performance highlights
                df_top_peak10s = filtered_df.nlargest(3, 'mdp_10')[[
                    'player_display', 'date', 'event_display', 'mdp_10', 'session_intensity_index'
                ]].copy()
                df_top_peak10s = df_top_peak10s.rename(columns={
                    'player_display': 'Player',
                    'date': 'Date',
                    'event_display': 'Session',
                    'mdp_10': 'Peak 10s (W)',
                    'session_intensity_index': 'Session intensity (z)'
                })
                df_top_peak10s = df_top_peak10s.round({'Peak 10s (W)': 0, 'Session intensity (z)': 2})
                
                if not df_top_peak10s.empty:
                    tmp = df_top_peak10s.copy()
                    tmp['Highlight category'] = "Most explosive"
                    highlight_frames.append(tmp)
            
            if len(highlight_frames) == 0:
                # Build all four tables fresh
                df_top_peak10s = filtered_df.nlargest(3, 'mdp_10')[[
                    'player_id', 'date', 'session_id', 'mdp_10', 'session_intensity_index'
                ]].copy()
                df_top_peak10s = df_top_peak10s.rename(columns={
                    'player_id': 'Player', 'date': 'Date', 'session_id': 'Session',
                    'mdp_10': 'Peak 10s (W)', 'session_intensity_index': 'Session intensity (z)'
                })
                df_top_peak10s = df_top_peak10s.round({'Peak 10s (W)': 0, 'Session intensity (z)': 2})
                
                if not df_top_peak10s.empty:
                    tmp = df_top_peak10s.copy()
                    tmp['Highlight category'] = "Most explosive"
                    highlight_frames.append(tmp)
                
                df_sustained_temp = filtered_df.copy()
                df_sustained_temp['sustained_avg'] = 0.5 * (df_sustained_temp['mdp_20'] + df_sustained_temp['mdp_30'])
                df_top_sustained = df_sustained_temp.nlargest(3, 'sustained_avg')[[
                    'player_display', 'date', 'event_display', 'mdp_20', 'session_intensity_index'
                ]].copy()
                df_top_sustained = df_top_sustained.rename(columns={
                    'player_display': 'Player', 'date': 'Date', 'event_display': 'Session',
                    'mdp_20': 'Peak 20s (W)', 'session_intensity_index': 'Session intensity (z)'
                })
                df_top_sustained = df_top_sustained.round({'Peak 20s (W)': 0, 'Session intensity (z)': 2})
                
                if not df_top_sustained.empty:
                    tmp = df_top_sustained.copy()
                    tmp['Highlight category'] = "Best sustained"
                    highlight_frames.append(tmp)
                
                df_top_load = filtered_df.nlargest(3, 'total_mp_load')[[
                    'player_display', 'date', 'event_display', 'total_mp_load', 'session_intensity_index'
                ]].copy()
                df_top_load = df_top_load.rename(columns={
                    'player_display': 'Player', 'date': 'Date', 'event_display': 'Session',
                    'total_mp_load': 'Total load (A.U.)', 'session_intensity_index': 'Session intensity (z)'
                })
                df_top_load = df_top_load.round({'Total load (A.U.)': 0, 'Session intensity (z)': 2})
                
                if not df_top_load.empty:
                    tmp = df_top_load.copy()
                    tmp['Highlight category'] = "Biggest workloads"
                    highlight_frames.append(tmp)
                
                if 'mdp_peak_value' in filtered_df.columns:
                    df_top_mdp_peak = filtered_df.nlargest(3, 'mdp_peak_value')[[
                        'player_display', 'date', 'event_display', 'mdp_peak_value', 'session_intensity_index'
                    ]].copy()
                    df_top_mdp_peak = df_top_mdp_peak.rename(columns={
                        'player_display': 'Player', 'date': 'Date', 'event_display': 'Session',
                        'mdp_peak_value': 'MDP peak (W)', 'session_intensity_index': 'Session intensity (z)'
                    })
                    df_top_mdp_peak = df_top_mdp_peak.round({'MDP peak (W)': 0, 'Session intensity (z)': 2})
                    
                    if not df_top_mdp_peak.empty:
                        tmp = df_top_mdp_peak.copy()
                        tmp['Highlight category'] = "Hardest sessions"
                        highlight_frames.append(tmp)
            
            if highlight_frames:
                highlights_export = pd.concat(highlight_frames, ignore_index=True)
                csv_highlights = highlights_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download highlight sessions (CSV)",
                    data=csv_highlights,
                    file_name="sessions_highlights_view.csv",
                    mime="text/csv",
                    key="download_sessions_highlights"
                )
            else:
                st.caption("No highlight rows available to export for this view.")


# ============================================================================
# MAIN PAGE EXECUTION
# ============================================================================



if not st.session_state.get('data_loaded', False) or st.session_state['raw_df'] is None:
    st.warning("⚠️ Please load data on the **Home** page first.")
    st.stop()

# Load session-level data with current weights
raw_df = st.session_state['raw_df'].copy()

# Ensure date is datetime
if 'date' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['date']):
    raw_df['date'] = pd.to_datetime(raw_df['date'])

with st.spinner("Computing session intensity metrics..."):
    session_df = get_session_intensity_df(
        raw_df,
        w_explosiveness=st.session_state['w_explosiveness'],
        w_repeatability=st.session_state['w_repeatability'],
        w_volume=st.session_state['w_volume']
    )

st.markdown("---")

# Call helper functions in sequence
filtered_df = render_filters_and_presets(session_df, view_mode)

# Early guard: if no sessions match filters, stop here
if filtered_df.empty:
    st.info("🤔 No sessions match your filters. Try adjusting date range, player selection, or intensity threshold above.")
    st.stop()

# Build display name mappings for use in both Coach and Analyst views
# This ensures consistent coach-friendly labels everywhere in the app
player_display_map = build_player_display_map(filtered_df, player_col='player_id')
filtered_df = add_player_display_column(filtered_df, player_display_map)
filtered_df = add_event_display_column(filtered_df)

st.markdown("---")

# Render backbone sections (shown in both Coach and Analyst modes)
render_summary_metrics_and_narrative(filtered_df)

st.markdown("---")

render_sessions_summary_table(filtered_df, view_mode)

st.markdown("---")

render_intensity_over_time_section(filtered_df, view_mode)

st.markdown("---")

# Analyst-only sections
render_regression_and_interpretation(filtered_df, view_mode)

render_intensity_distribution_histogram(filtered_df, view_mode)

render_player_load_overview(filtered_df, view_mode)

render_advanced_metrics_section(filtered_df, view_mode)

st.markdown("---")

render_performance_highlights_and_risk_radar(filtered_df, view_mode)

st.markdown("---")

render_view_exports_section(filtered_df, view_mode)

st.markdown("---")
st.caption("Sessions Explorer | Grow Irish Performance Analytics")

# Sidebar: Show current weighting configuration
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.caption(f"""
**Current Intensity Weights:**
- Explosiveness: {st.session_state['w_explosiveness']:.0%}
- Repeatability: {st.session_state['w_repeatability']:.0%}
- Volume: {st.session_state['w_volume']:.0%}

To adjust, visit the **Configuration** page.
""")
