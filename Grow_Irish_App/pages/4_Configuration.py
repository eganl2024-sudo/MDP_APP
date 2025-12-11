"""
Configuration Page - Intensity Weights, Data Management & Documentation

Manage global settings for intensity calculations and access documentation.
"""

import streamlit as st
from src.ui.nav import render_global_nav

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    st.session_state['raw_df'] = None

if 'w_explosiveness' not in st.session_state:
    st.session_state['w_explosiveness'] = 0.30
    st.session_state['w_repeatability'] = 0.50
    st.session_state['w_volume'] = 0.20

st.set_page_config(page_title="Configuration", layout="wide")
render_global_nav(current_page="config")
st.title("⚡ Configuration & Settings")

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Please load data on the **Home** page first.")
    st.stop()

# Section 1: Intensity Weights
st.markdown("## 📊 Intensity Weight Presets")

st.markdown("""
The **Session Intensity Index** combines three metrics with configurable weights:
- **Explosiveness (30%)** – Peak power in 10 seconds (MDP 10s)
- **Repeatability (50%)** – Sustained power (average of MDP 20s and MDP 30s)
- **Volume (20%)** – Total metabolic load accumulated in session

Choose a preset below or customize weights manually.
""")

# Display current settings
st.markdown("### Current Settings")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Explosiveness", f"{st.session_state['w_explosiveness']:.0%}")
with col2:
    st.metric("Repeatability", f"{st.session_state['w_repeatability']:.0%}")
with col3:
    st.metric("Volume", f"{st.session_state['w_volume']:.0%}")

# Preset buttons
st.markdown("### Select Preset")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Match-Like (30/50/20)", use_container_width=True, type="primary"):
        st.session_state['w_explosiveness'] = 0.30
        st.session_state['w_repeatability'] = 0.50
        st.session_state['w_volume'] = 0.20
        st.success("✅ Applied Match-Like preset")
        st.rerun()

with col2:
    if st.button("⚡ Speed Emphasis (40/35/25)", use_container_width=True):
        st.session_state['w_explosiveness'] = 0.40
        st.session_state['w_repeatability'] = 0.35
        st.session_state['w_volume'] = 0.25
        st.success("✅ Applied Speed Emphasis preset")
        st.rerun()

with col3:
    if st.button("💪 Conditioning (20/30/50)", use_container_width=True):
        st.session_state['w_explosiveness'] = 0.20
        st.session_state['w_repeatability'] = 0.30
        st.session_state['w_volume'] = 0.50
        st.success("✅ Applied Conditioning preset")
        st.rerun()

with st.expander("Advanced: Custom Weights"):
    st.markdown("*(Custom weight sliders can be added here in future versions)*")
    st.info("Currently, weights are set via presets. Manual adjustment coming soon.")

# Section 2: Data Management
st.markdown("---")
st.markdown("## 📁 Data Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Data Status")
    if st.session_state['raw_df'] is not None:
        df = st.session_state['raw_df']
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        if 'player_id' in df.columns:
            st.write(f"**Players:** {df['player_id'].nunique()}")
        if 'date' in df.columns:
            st.write(f"**Date Range:** {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        st.error("No data loaded")

with col2:
    st.subheader("Actions")
    if st.button("🔄 Reload Default Data", use_container_width=True):
        from utils import load_default_data
        try:
            raw_df = load_default_data("full_players_df.csv")
            st.session_state['raw_df'] = raw_df
            st.session_state['data_loaded'] = True
            st.session_state['data_source'] = 'default'
            st.success("✅ Reloaded default data")
            st.rerun()
        except FileNotFoundError:
            st.error("❌ Could not find full_players_df.csv")
    
    if st.button("📤 Go to Home to Upload", use_container_width=True):
        st.switch_page("pages/1_🏠_Home.py")

# Section 3: Documentation
st.markdown("---")
st.markdown("## 📚 Documentation & References")

with st.expander("What is Session Intensity Index?", expanded=True):
    st.markdown("""
    The **Session Intensity Index** is a standardized z-score that measures how demanding 
    a training session was compared to the team average.
    
    **Interpretation:**
    - **0** = Typical session (average difficulty)
    - **> 1** = Hard session (above average, demanding)
    - **< -1** = Light/Recovery session (below average, easy)
    
    **Formula:**
    ```
    Index = 0.30 × Explosiveness_z + 0.50 × Repeatability_z + 0.20 × Volume_z
    ```
    
    Where:
    - `Explosiveness_z` = Z-score of peak 10-second power (MDP 10s)
    - `Repeatability_z` = Z-score of average 20s & 30s peaks
    - `Volume_z` = Z-score of total metabolic load
    """)

with st.expander("Metabolic Power (MP)"):
    st.markdown("""
    **Metabolic Power** is calculated from tracking data using physics equations:
    - Input: Speed, acceleration, cadence
    - Output: Energy demand (watts)
    
    Used to compute:
    - **MDP (Most Demanding Period)**: Peak sustained power in windows (10s, 20s, 30s)
    - **Total Load**: Energy accumulated over entire session
    - **Intensity Components**: Explosiveness, Repeatability, Volume
    """)

with st.expander("How to Use Each Page"):
    st.markdown("""
    **📊 Sessions Page:**
    - View all sessions in a filterable table
    - Explore trends with interactive charts
    - Export data for external analysis
    
    **👥 Players Page:**
    - See individual player rolling window data
    - Compare intensity windows (5s, 10s, 20s, 30s)
    - Understand pacing and effort distribution
    
    **📚 Documentation Page:**
    - Learn about the system methodology
    - Access full technical reference
    - View research background
    """)

with st.expander("Weight Preset Details"):
    st.markdown("""
    **🎯 Match-Like (Default: 30/50/20)**
    - Emphasizes **sustained power** (50% on repeatability)
    - Balances peak and volume
    - Best for: Overall intensity assessment
    - Use when: Evaluating session difficulty
    
    **⚡ Speed Emphasis (40/35/25)**
    - Emphasizes **peak power** (40% on explosiveness)
    - Slightly lower sustained emphasis
    - Best for: High-speed training blocks
    - Use when: Speed work is the focus
    
    **💪 Conditioning (20/30/50)**
    - Emphasizes **volume/load** (50% on volume)
    - Minimizes peak power importance
    - Best for: Aerobic/endurance sessions
    - Use when: Volume training is the focus
    """)

st.markdown("---")
st.info("""
**Need Help?**
- Hover over metrics and charts for detailed tooltips
- See **Documentation** page for full system overview
- Contact your coach or analyst for interpretation help
""")
