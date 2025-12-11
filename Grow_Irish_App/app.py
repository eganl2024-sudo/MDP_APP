"""
Grow Irish Session Intensity & MDP Explorer - Multi-Page App

Main entry point for the Streamlit application.
Initializes session state, configures pages, and sets up navigation.
"""

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Grow Irish Performance Analytics",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    st.session_state['raw_df'] = None
    st.session_state['session_df'] = None
    st.session_state['hr_df'] = None
    st.session_state['data_source'] = 'default'

if 'w_explosiveness' not in st.session_state:
    st.session_state['w_explosiveness'] = 0.30  # Default: Match-like
    st.session_state['w_repeatability'] = 0.50
    st.session_state['w_volume'] = 0.20

if 'selected_players' not in st.session_state:
    st.session_state['selected_players'] = None

if 'date_range' not in st.session_state:
    st.session_state['date_range'] = None

if 'high_intensity_only' not in st.session_state:
    st.session_state['high_intensity_only'] = False

# Sidebar Navigation
st.sidebar.markdown("# 🏃 Grow Irish Analytics")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📊 ANALYSIS")
with st.sidebar:
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.switch_page("pages/1_Home.py")
    
    if st.button("📊 Sessions", use_container_width=True, key="nav_sessions"):
        st.switch_page("pages/2_Sessions.py")
    
    if st.button("👥 Players", use_container_width=True, key="nav_players"):
        st.switch_page("pages/3_Players.py")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ TOOLS & REFERENCE")
with st.sidebar:
    if st.button("⚡ Configuration", use_container_width=True, key="nav_config"):
        st.switch_page("pages/4_Configuration.py")
    
    if st.button("📚 Documentation", use_container_width=True, key="nav_docs"):
        st.switch_page("pages/5_Documentation.py")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Status**: Data loaded ✅" if st.session_state['data_loaded'] 
    else "**Status**: No data loaded ⏳"
)

# Main area for home page content
st.title("🏃 Grow Irish Session Intensity & MDP Explorer")

st.markdown("""
Welcome to the Grow Irish Performance Analytics platform. This app provides comprehensive 
analysis of player session intensity, metabolic power demands, and peak performance metrics.

**Quick Navigation:**
- 📊 **Sessions** – Explore all sessions, filter by player/date, visualize trends
- 👥 **Players** – Individual player profiles with rolling window analysis
- ⚡ **Configuration** – Adjust intensity weights and manage data
- 📚 **Documentation** – Learn about the system and methodology

**Start Here:**
1. Go to **Home** page to upload data or use defaults
2. Explore **Sessions** to see overall patterns
3. Dive into **Players** for individual analysis
4. Adjust weights on **Configuration** if needed
""")

if not st.session_state['data_loaded']:
    st.warning("👉 **No data loaded yet.** Please go to the **Home** page to upload or load default data.")
else:
    st.success(f"✅ **Data loaded!** ({st.session_state.get('data_source', 'unknown')})")
    st.info(f"""
    **Current Configuration:**
    - Explosiveness: {st.session_state['w_explosiveness']:.0%}
    - Repeatability: {st.session_state['w_repeatability']:.0%}
    - Volume: {st.session_state['w_volume']:.0%}
    
    [Edit on Configuration page →]
    """)
