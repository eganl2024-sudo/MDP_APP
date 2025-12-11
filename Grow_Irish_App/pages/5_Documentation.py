"""
Documentation Page - System Overview, Methodology & References

Complete documentation and technical reference for the analytics platform.
"""

import streamlit as st
from src.ui.nav import render_global_nav

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

st.set_page_config(page_title="Documentation", layout="wide")
render_global_nav(current_page="docs")
st.title("📚 Documentation & Reference")

st.markdown("""
This page provides comprehensive documentation about the Grow Irish Session Intensity 
& MDP Explorer analytics platform.
""")

# System Overview
with st.expander("📖 System Overview", expanded=True):
    st.markdown("""
    ## What is This System?
    
    The **Session Intensity & MDP Explorer** is a complete performance analytics platform 
    for soccer players. It combines:
    
    1. **Metabolic Power (MP) Computation** – Physics-based calculation of energy demand from GPS tracking
    2. **Session Aggregation** – Converts raw tracking data to session-level summaries
    3. **Intensity Scoring** – Combines explosiveness, repeatability, and volume into a single metric
    4. **Interactive Exploration** – Visualize trends, filter data, and export results
    
    ### Key Features
    - **Real-time computation**: Load data and instantly get intensity metrics
    - **Flexible weighting**: Adjust intensity formula with 3 preset profiles
    - **Multi-level analysis**: Session-level, player-level, and team-level insights
    - **Export-ready**: Download data in CSV for further analysis
    - **Coach-friendly**: Clear labels, helpful tooltips, visual references
    """)

with st.expander("🔬 Methodology"):
    st.markdown("""
    ## How Intensity is Calculated
    
    ### Step 1: Metabolic Power (MP)
    From GPS tracking data (speed, acceleration, cadence), we compute MP using a validated equation:
    
    ```
    MP = 1.99 + 26.70×speed - 0.18×speed² - 14.96×a_pos - 21.75×a_neg 
         + 23.42×(a_pos×speed) - 2.56×(a_neg×speed) - 0.017×cadence
    ```
    
    Where:
    - `a_pos` = positive acceleration (speeding up)
    - `a_neg` = negative acceleration (slowing down)
    - Result: Energy demand in watts
    
    ### Step 2: Most Demanding Periods (MDP)
    Using rolling windows, we find peak sustained power:
    - **MDP 10s** = Highest average power in any 10-second window
    - **MDP 20s** = Highest average power in any 20-second window
    - **MDP 30s** = Highest average power in any 30-second window
    
    ### Step 3: Session Components
    For each session, compute three metrics:
    - **Explosiveness** = MDP 10s (peak power)
    - **Repeatability** = Average of MDP 20s and MDP 30s (sustained power)
    - **Volume** = Total MP × session duration (cumulative load)
    
    ### Step 4: Normalization (Z-scores)
    Each component is normalized across the team to account for differences:
    ```
    Component_z = (Component - Team_Mean) / Team_StdDev
    ```
    
    ### Step 5: Intensity Index
    Final score combines components with configurable weights:
    ```
    Index = w_explosiveness × Explosiveness_z 
          + w_repeatability × Repeatability_z 
          + w_volume × Volume_z
    ```
    
    **Default weights (Match-Like)**: 30% Explosiveness, 50% Repeatability, 20% Volume
    
    ### Result Interpretation
    - **0** = Team average intensity
    - **> 1** = Above average (hard session)
    - **< -1** = Below average (easy/recovery session)
    """)

with st.expander("🎯 Understanding Weight Presets"):
    st.markdown("""
    ### Match-Like (Default: 30/50/20)
    **Best for**: Overall intensity assessment
    
    - **Explosiveness 30%**: Peak power is important
    - **Repeatability 50%**: Sustained power is most important (mirrors game demands)
    - **Volume 20%**: Total load matters but secondary
    
    **When to use**: General training analysis, assessing session difficulty
    
    ---
    
    ### Speed Emphasis (40/35/25)
    **Best for**: Speed and sprint-focused training
    
    - **Explosiveness 40%**: Peak power is heavily weighted
    - **Repeatability 35%**: Sustained power slightly reduced
    - **Volume 25%**: Load is less emphasized
    
    **When to use**: Speed work blocks, sprint sessions, acceleration development
    
    ---
    
    ### Conditioning (20/30/50)
    **Best for**: Aerobic and volume-based training
    
    - **Explosiveness 20%**: Peak power minimized
    - **Repeatability 30%**: Moderate sustained power importance
    - **Volume 50%**: Total workload is the focus
    
    **When to use**: Endurance blocks, aerobic conditioning, high-volume sessions
    """)

with st.expander("📊 Using Each Page"):
    st.markdown("""
    ## 🏠 Home Page
    **Purpose**: Load data and understand the system
    
    - Upload CSV files or load default data
    - View data summary (rows, columns, date range)
    - Read quick start guide
    - Understand how to use the app in 5 minutes
    
    **Key Actions**:
    - Load default data (full_players_df.csv) with 48 pre-computed sessions
    - Upload custom tracking data with speed, acceleration, HR, MP
    - Download loaded data as CSV
    
    ---
    
    ## 📊 Sessions Page
    **Purpose**: Explore all sessions, trends, and team-level patterns
    
    - Filter by player, date range, intensity level
    - See table of all sessions ranked by intensity
    - View interactive charts (intensity over time, load vs intensity)
    - Export filtered data for analysis
    
    **Key Visualizations**:
    - **Intensity Over Time** – Trend line showing session progression
    - **Load vs Intensity** – Scatter showing relationship between work and difficulty
    - **Intensity Distribution** – Histogram of how many sessions at each intensity
    - **MDP Comparison** – Bar chart comparing peak power windows
    
    **Key Metrics**:
    - Sessions in View (filtered count)
    - Average Session Intensity (team mean)
    - Most Intense Session (highest z-score)
    - Average Peak 10s Power (in watts)
    
    ---
    
    ## 👥 Players Page
    **Purpose**: Analyze individual player effort and pacing
    
    - Select specific session and player
    - Compare rolling window intensities (5s, 10s, 20s, 30s)
    - See how intensity varies over time within a session
    - Understand player pacing and effort distribution
    
    **Key Visualization**:
    - **Rolling Window Chart** – Shows intensity smoothed over different time windows
    - Helps identify pacing patterns (e.g., fast start, steady, late push)
    
    ---
    
    ## ⚡ Configuration Page
    **Purpose**: Manage settings and learn about the system
    
    - View current intensity weight preset
    - Switch between 3 presets (Match-Like, Speed, Conditioning)
    - Reload or upload new data
    - Read full documentation and definitions
    
    **Key Actions**:
    - Change weight preset (automatically recalculates all metrics)
    - Reload default data
    - Access documentation about each preset
    """)

with st.expander("📖 Technical Reference"):
    st.markdown("""
    ## Data Requirements
    
    ### For Session-Level Analysis (Recommended)
    You need a CSV with session-level data already computed:
    ```
    player_id, date, session_intensity_index, mdp_10, mdp_20, mdp_30, 
    total_mp_load, session_duration_s, explosiveness_z, repeatability_z, volume_z
    ```
    
    **Example**: full_players_df.csv (48 sessions × 18 columns)
    
    ### For Player-Level Rolling Window Analysis
    You need raw tracking data with samples at frequent intervals:
    ```
    timestamp, player_id, date, speed, acceleration, hr, mp, cadence, [other cols]
    ```
    
    **Note**: Timestamp should be milliseconds since epoch or ISO format
    
    ---
    
    ## Computation Time
    
    | Operation | Time |
    |-----------|------|
    | Load 48 sessions | <1 second (cached) |
    | Change weights | <1 second (all pages auto-update) |
    | Session filters | <1 second |
    | Rolling windows (48 sessions) | 5-10 seconds (first run, then cached) |
    | Chart rendering | <1 second |
    
    ---
    
    ## File Structure
    
    ```
    Grow Irish/
    ├── app.py                    ← Main entry point
    ├── utils.py                  ← Shared utilities
    ├── mp_intensity_pipeline.py  ← Backend (MP calculation)
    ├── full_players_df.csv       ← Default session data
    └── pages/
        ├── 1_🏠_Home.py
        ├── 2_📊_Sessions.py
        ├── 3_👥_Players.py
        ├── 4_⚡_Configuration.py
        └── 5_📚_Documentation.py
    ```
    """)

with st.expander("❓ FAQ"):
    st.markdown("""
    ### Q: What does a Session Intensity Index of 1.5 mean?
    **A**: The session was 1.5 standard deviations above the team average in intensity.
    It was a hard, demanding session – significantly above typical difficulty.
    
    ---
    
    ### Q: Why do intensity scores change when I switch presets?
    **A**: The intensity formula has three components (Explosiveness, Repeatability, Volume).
    Each preset weighs them differently. Switching presets recalculates all metrics.
    
    ---
    
    ### Q: Can I use custom weights instead of presets?
    **A**: Currently, only the 3 presets are available. Custom weight sliders are planned for a future version.
    
    ---
    
    ### Q: What's the difference between MDP 10s, 20s, and 30s?
    **A**: 
    - **MDP 10s** = Maximum peak power (quick, explosive efforts)
    - **MDP 20s** = Sustained power for 20 seconds (typical sprint duration)
    - **MDP 30s** = Sustained power for 30 seconds (longer efforts, anaerobic capacity)
    
    ---
    
    ### Q: How is "Volume" calculated?
    **A**: Volume = Mean MP × Session Duration
    
    It represents total energy accumulated over the entire session.
    
    ---
    
    ### Q: Can I export data for further analysis?
    **A**: Yes! On the Sessions page, there's a download button that exports all metrics as CSV.
    You can then use Excel, Python, R, or other tools for deeper analysis.
    
    ---
    
    ### Q: What if my data is missing some columns?
    **A**: The app will tell you which columns are missing. 
    Required: player_id, date, speed, acceleration, cadence, hr, mp
    
    ---
    
    ### Q: How accurate is the MP calculation?
    **A**: The metabolic power formula was validated against real player data with R² > 0.97.
    It's a highly accurate physics-based model.
    """)

with st.expander("🔗 External Resources"):
    st.markdown("""
    - **MASTER_SUMMARY.md** – Complete system overview and project history
    - **SESSION_INTENSITY_EXPLORER_GUIDE.md** – Detailed coach-friendly guide
    - **APPLICATIONS_GUIDE.md** – How to integrate into other systems
    - **mp_intensity_pipeline.py** – Backend source code documentation
    
    For more information, see the project documentation files.
    """)

st.markdown("---")
st.info("""
**Questions or Feedback?**

Contact your analyst or coach for help interpreting results.

For technical issues, refer to this documentation or the project's MASTER_SUMMARY.md file.
""")

st.caption("Documentation | Grow Irish Performance Analytics | December 2025")
