"""
Display Names Module - Shared utilities for coach-friendly labels

Single source of truth for:
- Player display mapping (Player 01, Player 02, ...)
- Event/session display labels (Player01_event_MM-DD-YYYY)

Used by Sessions Explorer, Player Analysis, and other pages.
"""

import pandas as pd
import numpy as np


def build_player_display_map(df: pd.DataFrame, player_col: str = "player_id") -> dict:
    """
    Build mapping from internal player IDs to coach-friendly display labels.
    
    Args:
        df: DataFrame containing player_col
        player_col: Column name with player IDs (default: "player_id")
    
    Returns:
        Dict mapping internal IDs to labels: {pid_1: "Player 01", pid_2: "Player 02", ...}
    """
    unique_players = sorted(df[player_col].dropna().unique())
    mapping = {}
    for idx, pid in enumerate(unique_players, start=1):
        mapping[pid] = f"Player {idx:02d}"
    return mapping


def add_player_display_column(df: pd.DataFrame, player_display_map: dict) -> pd.DataFrame:
    """
    Add player_display column by mapping player_id using provided mapping dict.
    
    Args:
        df: Input DataFrame
        player_display_map: Dict from build_player_display_map()
    
    Returns:
        DataFrame with new player_display column
    """
    df = df.copy()
    df["player_display"] = df["player_id"].map(player_display_map)
    return df


def add_event_display_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coach-friendly event/session label: 'Player01_event_MM-DD-YYYY'.
    
    Assumes player_display column already exists and date column is present.
    
    Args:
        df: Input DataFrame with player_display and date columns
    
    Returns:
        DataFrame with new event_display column
    
    Raises:
        ValueError: If player_display column missing
    """
    df = df.copy()
    
    if "player_display" not in df.columns:
        raise ValueError("player_display column missing; run add_player_display_column first.")
    
    # Compact player tag: remove space from 'Player 01' -> 'Player01'
    df["player_tag"] = df["player_display"].str.replace(" ", "", regex=False)
    
    # Format date as MM-DD-YYYY
    # Try common date column names
    date_col = None
    for col_name in ["date", "session_date"]:
        if col_name in df.columns:
            date_col = col_name
            break
    
    if date_col:
        if np.issubdtype(df[date_col].dtype, np.datetime64):
            date_str = df[date_col].dt.strftime("%m-%d-%Y")
        else:
            date_str = pd.to_datetime(df[date_col]).dt.strftime("%m-%d-%Y")
        df["event_display"] = df["player_tag"] + "_event_" + date_str
    else:
        # Fallback: no event display without date
        df["event_display"] = df["player_tag"] + "_event_unknown"
    
    return df
