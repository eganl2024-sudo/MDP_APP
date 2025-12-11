"""
Intensity Classification Module - Shared utility for coach-friendly intensity labels

Maps intensity percentiles (0-100) to categorical labels:
- Easy (< 25th percentile)
- Medium (25-60th percentile)
- Hard (60-85th percentile)
- Very Hard (≥ 85th percentile)

Used by Player Analysis and other pages for consistent intensity categorization.
"""

import pandas as pd
from typing import Optional


def classify_intensity_from_percentile(percentile: Optional[float]) -> str:
    """
    Map an intensity percentile (0-100) to a coach-facing label.
    
    Args:
        percentile: Float value 0-100, or NaN/None for unknown
    
    Returns:
        Category label: "Easy", "Medium", "Hard", "Very Hard", or "Unknown"
    
    Examples:
        classify_intensity_from_percentile(10)   -> "Easy"
        classify_intensity_from_percentile(45)   -> "Medium"
        classify_intensity_from_percentile(70)   -> "Hard"
        classify_intensity_from_percentile(90)   -> "Very Hard"
        classify_intensity_from_percentile(None) -> "Unknown"
    """
    if percentile is None or pd.isna(percentile):
        return "Unknown"
    
    if percentile < 25:
        return "Easy"
    elif percentile < 60:
        return "Medium"
    elif percentile < 85:
        return "Hard"
    else:
        return "Very Hard"


def classify_explosiveness(percentile: Optional[float]) -> str:
    """
    Map a peak/explosiveness percentile (0-100) to a coach-facing label.
    
    Uses same thresholds as intensity classification for consistency.
    
    Args:
        percentile: Float value 0-100, or NaN/None for unknown
    
    Returns:
        Category label: "Low", "Moderate", "High", "Very high", or "Unknown"
    
    Examples:
        classify_explosiveness(10)   -> "Low"
        classify_explosiveness(45)   -> "Moderate"
        classify_explosiveness(70)   -> "High"
        classify_explosiveness(90)   -> "Very high"
        classify_explosiveness(None) -> "Unknown"
    """
    if percentile is None or pd.isna(percentile):
        return "Unknown"
    
    if percentile < 25:
        return "Low"
    elif percentile < 60:
        return "Moderate"
    elif percentile < 85:
        return "High"
    else:
        return "Very high"
