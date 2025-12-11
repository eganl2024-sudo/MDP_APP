"""
Configuration Module - Coach-friendly labels and mappings

Central configuration for:
- Intensity window labels (internal keys → coach-friendly names)
- Window colors for consistent visualization
- Other shared configuration constants

Used app-wide for consistent UX labels and colors.
"""

# ============================================================================
# INTENSITY WINDOW LABELS
# ============================================================================
# Map from internal database keys to coach-friendly display labels
# Used in dropdowns, charts, summaries, and exports

INTENSITY_WINDOW_LABELS = {
    "intensity_5s": "Burst (5s)",
    "intensity_10s": "Burst (10s)",
    "intensity_20s": "Short press (20s)",
    "intensity_30s": "Extended press (30s)",
    "intensity_60s": "Sustained phase (60s)",
    "intensity_180s": "Long phase (180s)",
}

# ============================================================================
# WINDOW COLORS - High contrast palette for visualization
# ============================================================================
# Maps internal window names to hex colors for consistent, distinguishable rendering
# Used in Intensity Over Time charts and all multi-window visualizations

WINDOW_COLOR_MAP = {
    # Original internal naming (if traces use this format)
    "5s window": "#d62728",          # red
    "10s window": "#1f77b4",         # blue
    "20s window": "#2ca02c",         # green
    "30s window": "#ff7f0e",         # orange
    "60s window": "#9467bd",         # purple
    "180s window": "#8c564b",        # brown
    
    # Coach-friendly naming (if traces use friendly labels)
    "Burst (5s)": "#d62728",         # red
    "Burst (10s)": "#1f77b4",        # blue
    "Short press (20s)": "#2ca02c",  # green
    "Extended press (30s)": "#ff7f0e",  # orange
    "Sustained phase (60s)": "#9467bd",  # purple
    "Long phase (180s)": "#8c564b",  # brown
    
    # Also support the raw internal keys for flexibility
    "intensity_5s": "#d62728",       # red
    "intensity_10s": "#1f77b4",      # blue
    "intensity_20s": "#2ca02c",      # green
    "intensity_30s": "#ff7f0e",      # orange
    "intensity_60s": "#9467bd",      # purple
    "intensity_180s": "#8c564b",     # brown
}

# Window durations in seconds (for calculations)
WINDOW_SECONDS = {
    "intensity_5s": 5,
    "intensity_10s": 10,
    "intensity_20s": 20,
    "intensity_30s": 30,
    "intensity_60s": 60,
    "intensity_180s": 180,
}

# Default intensity windows to display in Coach view (non-configurable)
DEFAULT_COACH_WINDOWS = [
    "intensity_10s",
    "intensity_20s",
    "intensity_30s",
]

# Default intensity windows in Analyst view selector
DEFAULT_ANALYST_WINDOWS = [
    "intensity_10s",
    "intensity_20s",
]

# ============================================================================
# OTHER SHARED CONSTANTS
# ============================================================================

# Available windows for Analyst view multiselect
AVAILABLE_INTENSITY_WINDOWS = [
    "intensity_5s",
    "intensity_10s",
    "intensity_20s",
    "intensity_30s",
]

# Reverse mapping for conversion from labels back to keys (for internal use)
def get_key_from_label(label: str) -> str:
    """
    Convert coach-friendly label back to internal key.
    
    Args:
        label: Coach-friendly label (e.g., "Burst (10s)")
    
    Returns:
        Internal key (e.g., "intensity_10s"), or original label if not found
    """
    for key, label_val in INTENSITY_WINDOW_LABELS.items():
        if label_val == label:
            return key
    return label  # Fallback: assume it's already a key


def get_label_from_key(key: str) -> str:
    """
    Convert internal key to coach-friendly label.
    
    Args:
        key: Internal key (e.g., "intensity_10s")
    
    Returns:
        Coach-friendly label (e.g., "Burst (10s)"), or original key if not found
    """
    return INTENSITY_WINDOW_LABELS.get(key, key)  # Fallback: return key unchanged


def get_window_color(window_name: str) -> str:
    """
    Get color for a window by name (supports multiple naming conventions).
    
    Args:
        window_name: Window name (e.g., "5s window", "intensity_10s", "Burst (10s)")
    
    Returns:
        Hex color code, or default gray if not found
    """
    return WINDOW_COLOR_MAP.get(window_name, "#333333")  # Default to dark gray if not found
