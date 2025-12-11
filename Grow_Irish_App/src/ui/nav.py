"""
Global Navigation Helper - Minimal horizontal nav bar for multi-page Streamlit app.

Renders at the top of every page, showing links to all pages with current page bold.
"""

import streamlit as st


def render_global_nav(current_page: str) -> None:
    """
    Render a minimal horizontal nav bar at the top of the page.

    Parameters
    ----------
    current_page : str
        One of: 'home', 'sessions', 'players', 'config', 'docs'
        Determines which link is displayed bold.
    """
    st.markdown("---")

    # Create columns with relative widths
    col_home, col_sessions, col_players, col_config, col_docs = st.columns(
        [1, 1, 1.2, 1.6, 1]
    )

    def label(text: str, key: str) -> str:
        """Wrap label in bold if it matches the current page."""
        return f"**{text}**" if current_page == key else text

    with col_home:
        st.page_link(
            "pages/1_Home.py",
            label=label("Home", "home"),
        )

    with col_sessions:
        st.page_link(
            "pages/2_Sessions.py",
            label=label("Sessions", "sessions"),
        )

    with col_players:
        st.page_link(
            "pages/3_Players.py",
            label=label("Players", "players"),
        )

    with col_config:
        st.page_link(
            "pages/4_Configuration.py",
            label=label("Configuration", "config"),
        )

    with col_docs:
        st.page_link(
            "pages/5_Documentation.py",
            label=label("Docs", "docs"),
        )

    st.markdown("")  # Small vertical spacer
