"""
Home Page - Data Loading & Quick Start Guide

Upload or load default data, view summary, and learn how to use the app.
"""

import streamlit as st
import pandas as pd
from utils import load_and_clean, load_default_data
from src.display_names import build_player_display_map, add_player_display_column, add_event_display_column
from src.ui.nav import render_global_nav

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    st.session_state['raw_df'] = None
    st.session_state['session_df'] = None
    st.session_state['hr_df'] = None

if 'w_explosiveness' not in st.session_state:
    st.session_state['w_explosiveness'] = 0.30
    st.session_state['w_repeatability'] = 0.50
    st.session_state['w_volume'] = 0.20

st.set_page_config(page_title="Home", layout="wide")
render_global_nav(current_page="home")
st.title("🏠 Home & Data Loading")
st.caption("Load your GPS data here and then explore team and player intensity across the app.")

# Data loading section
st.markdown("## 📥 Load Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Default Data")
    st.markdown("""
    Use the pre-loaded `full_players_df.csv` file with 48 sessions.
    """)
    if st.button("Load Default Data", type="primary", key="load_default"):
        try:
            with st.spinner("Loading default data..."):
                raw_df = load_default_data("full_players_df.csv")
                
                # Build and apply display name mapping (shared across all pages)
                player_display_map = build_player_display_map(raw_df, player_col="player_id")
                raw_df = add_player_display_column(raw_df, player_display_map)
                raw_df = add_event_display_column(raw_df)
                
                st.session_state['raw_df'] = raw_df
                st.session_state['player_display_map'] = player_display_map
                st.session_state['data_loaded'] = True
                st.session_state['data_source'] = 'default'
                st.success(f"✅ Loaded {len(raw_df)} rows of session data")
                st.rerun()
        except FileNotFoundError:
            st.error("❌ Could not find full_players_df.csv")

with col2:
    st.subheader("Option 2: Upload Custom CSV")
    st.markdown("""
    Upload one or more CSV files with tracking data (speed, acc, hr, mp, etc.)
    """)
    uploaded_files = st.file_uploader(
        "Choose CSV file(s)",
        type=['csv'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("Load Uploaded Data", type="primary", key="load_uploaded"):
            try:
                with st.spinner("Loading and cleaning data..."):
                    raw_df = load_and_clean(uploaded_files)
                    
                    # Build and apply display name mapping (shared across all pages)
                    player_display_map = build_player_display_map(raw_df, player_col="player_id")
                    raw_df = add_player_display_column(raw_df, player_display_map)
                    raw_df = add_event_display_column(raw_df)
                    
                    st.session_state['raw_df'] = raw_df
                    st.session_state['player_display_map'] = player_display_map
                    st.session_state['data_loaded'] = True
                    st.session_state['data_source'] = 'upload'
                    st.success(f"✅ Loaded and cleaned {len(raw_df)} rows from {len(uploaded_files)} file(s)")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")

st.caption("Select one option above and click Load to proceed. Data will be verified and ready to explore.")

st.markdown("---")

# Show data summary if loaded
if st.session_state['data_loaded'] and st.session_state['raw_df'] is not None:
    st.success("✅ Your data is loaded. You can now explore **Sessions Explorer** and **Player Analysis**.")
    
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview")
    
    raw_df = st.session_state['raw_df']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(raw_df):,}")
    with col2:
        st.metric("Total Columns", len(raw_df.columns))
    with col3:
        if 'player_id' in raw_df.columns:
            st.metric("Unique Players", raw_df['player_id'].nunique())
        else:
            st.metric("Players", "N/A")
    with col4:
        if 'date' in raw_df.columns:
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(raw_df['date']):
                raw_df['date'] = pd.to_datetime(raw_df['date'])
            st.metric("Date Range", f"{raw_df['date'].min().date()} to {raw_df['date'].max().date()}")
        else:
            st.metric("Dates", "N/A")
    
    # Date range badge with context
    if 'date' in raw_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_df['date']):
            raw_df['date'] = pd.to_datetime(raw_df['date'])
        min_date = raw_df['date'].min()
        max_date = raw_df['date'].max()
        num_days = (max_date - min_date).days + 1
        
        if num_days < 10:
            badge = "🔴"
        elif num_days < 14:
            badge = "🟡"
        else:
            badge = "🟢"
        
        st.caption(f"{badge} Training period: {min_date.date()} → {max_date.date()} ({num_days} days)")
    
    # Key columns validation
    st.markdown("---")
    required_cols = ["player_id", "date", "speed", "acc", "hr", "mp"]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Missing key columns: {', '.join(missing_cols)}. Some analysis features may not work as expected.")
    else:
        st.success("✓ All key columns for analysis are present.")
    
    st.markdown("---")
    
    # Data preview
    st.markdown("### 📋 Sample Data")
    st.dataframe(raw_df.head(20), height=350, use_container_width=True)
    st.caption("Preview of the first 20 rows from your dataset.")
    
    # Column info in expander
    with st.expander("Column definitions (click to expand)"):
        col_info = pd.DataFrame({
            'Column': raw_df.columns,
            'Type': raw_df.dtypes.values,
            'Non-Null': raw_df.notna().sum().values,
            'Null': raw_df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    # Missing values summary in expander
    with st.expander("Missing values summary"):
        missing = raw_df.isna().sum().to_frame("Missing count")
        st.dataframe(missing, use_container_width=True)
        st.caption("Counts of missing values for each column.")

else:
    st.info("Load default data or upload a CSV above to see a summary of your dataset.")

# Sidebar status indicator
st.sidebar.markdown("## ⚙️ Data Status")
st.sidebar.caption(f"**Status**: {'Loaded ✓' if st.session_state['data_loaded'] else 'Not loaded - please load data first'}")

st.markdown("---")
st.markdown("### How to use this app (5-minute guide)")

with st.expander("Step 1 – Load your data", expanded=False):
    st.markdown("""
    1. Choose **Option 1** (default dataset) or **Option 2** (upload your own CSV files).
    2. Click the **Load Default Data** button or confirm your upload.
    3. Wait for the green success message and check the **Dataset Overview** to confirm rows, columns, players, and date range.
    4. Use the **Sample Data** table and column definitions to verify that your key fields (time, player, MP, etc.) are present.
    """)

with st.expander("Step 2 – Explore sessions"):
    st.markdown("""
    1. Go to the **Sessions Explorer** page (sidebar).
    2. Select a date range and apply any filters you need (team, player count, intensity window).
    3. Use the charts to see how session intensity changes over time and to spot very light or very heavy training days.
    4. Export session-level rolling window data from this page if you want to do deeper analysis in Python or R.
    """)

with st.expander("Step 3 – Analyze players"):
    st.markdown("""
    1. Go to the **Player Analysis** page (sidebar).
    2. Pick a session from the dropdown, then select a player.
    3. Choose **Coach** view for simple tiles (Intensity Score, peak windows, total load, percentiles, early/mid/late trend).
    4. Switch to **Analyst** view to see the full intensity time series and detailed MDP summary for the selected intensity windows (e.g., 10s, 20s, 30s).
    5. Use the percentiles and light / moderate / hard / very hard labels to understand how demanding the session was for this player relative to the team.
    """)

with st.expander("Step 4 – Adjust intensity settings"):
    st.markdown("""
    1. Go to the **Configuration** page (sidebar).
    2. Review or change the preset intensity model:
       - **Match-like** (balanced across MDP, explosiveness, repeatability, and volume)
       - **Speed emphasis** (stronger weight on short peak windows and explosive efforts)
       - **Conditioning emphasis** (stronger weight on total volume and repeatability)
    3. If available, fine-tune the individual weights for MDP, explosiveness, repeatability, and volume.
    4. Save your settings; all pages (Sessions Explorer and Player Analysis) will automatically update to use the new intensity model.
    """)

with st.expander("Step 5 – Get help & documentation"):
    st.markdown("""
    1. Open the **Documentation** page from the sidebar for a full system overview and metric definitions.
    2. Hover over icons and metric labels throughout the app to see quick tooltips.
    3. If you are sharing this app with other staff, point them to this Home page for loading data and to this 5-minute guide for a quick orientation.
    """)

st.markdown("---")
st.caption("Home & Data Loading | Grow Irish Performance Analytics")
