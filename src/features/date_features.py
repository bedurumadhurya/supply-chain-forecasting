"""
Date Features Module

This module creates time-based features from date columns.
These features help models capture seasonality and temporal patterns.

Features created:
- day: Day of month (1-31)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- week_of_year: Week number (1-52)
- month: Month (1-12)
- quarter: Quarter (1-4)
- year: Year
- is_weekend: Binary indicator for Saturday/Sunday
"""

import pandas as pd
import numpy as np
from typing import List

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DateFeatureEngineer:
    """
    Engineer date-based features from datetime columns.
    
    These features are crucial for capturing:
    - Daily patterns (day of week effects)
    - Weekly seasonality (weekends vs weekdays)
    - Monthly seasonality (beginning/end of month)
    - Yearly seasonality (quarters, months)
    """
    
    def __init__(self):
        """Initialize the date feature engineer."""
        self.feature_names = [
            'day',
            'day_of_week',
            'week_of_year',
            'month',
            'quarter',
            'year',
            'is_weekend'
        ]
    
    def create_date_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Create all date-based features.
        
        Args:
            df: Input DataFrame with a date column
            date_column: Name of the date column
            
        Returns:
            DataFrame with additional date feature columns
            
        Note:
            The input DataFrame is not modified. A copy is returned with new features.
        """
        logger.info("Creating date features...")
        
        df = df.copy()
        
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract date components
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek  # Monday=0, Sunday=6
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['year'] = df[date_column].dt.year
        
        # Weekend indicator (Saturday=5, Sunday=6)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info(f"✓ Created {len(self.feature_names)} date features")
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encoding of date features.
        
        Cyclical encoding captures the circular nature of time:
        - Day 1 and day 31 are close in a cyclical sense
        - Monday (0) and Sunday (6) are adjacent
        
        This uses sine/cosine transformation:
        - day_of_week_sin = sin(2π * day_of_week / 7)
        - day_of_week_cos = cos(2π * day_of_week / 7)
        
        Args:
            df: DataFrame with date features
            
        Returns:
            DataFrame with additional cyclical features
            
        Note:
            This is particularly useful for neural network models.
        """
        logger.info("Creating cyclical date features...")
        
        df = df.copy()
        
        # Day of week cyclical features (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month cyclical features (1-31)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Month cyclical features (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        logger.info("✓ Created 6 cyclical date features")
        
        return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday and event features from M5 calendar data.
    
    The M5 dataset includes:
    - event_name_1, event_type_1
    - event_name_2, event_type_2
    - snap_CA, snap_TX, snap_WI (SNAP days by state)
    
    Args:
        df: DataFrame with M5 calendar columns
        
    Returns:
        DataFrame with processed holiday features
        
    Note:
        This assumes the calendar data has already been merged.
    """
    logger.info("Processing holiday and event features...")
    
    df = df.copy()
    
    # Check if event columns exist
    event_cols = [col for col in df.columns if col.startswith('event_')]
    snap_cols = [col for col in df.columns if col.startswith('snap_')]
    
    # Create binary indicator for any event
    if 'event_name_1' in df.columns:
        df['has_event'] = (~df['event_name_1'].isna()).astype(int)
        logger.info("✓ Created 'has_event' feature")
    
    # SNAP features are already binary (0/1)
    if snap_cols:
        logger.info(f"✓ Found {len(snap_cols)} SNAP features")
    
    # Event type indicators (if needed)
    if 'event_type_1' in df.columns:
        # Create dummy variables for event types
        event_types = df['event_type_1'].fillna('None')
        event_dummies = pd.get_dummies(event_types, prefix='event_type')
        df = pd.concat([df, event_dummies], axis=1)
        logger.info(f"✓ Created {len(event_dummies.columns)} event type indicators")
    
    logger.info("✓ Holiday and event features processed")
    
    return df


def get_date_feature_names(include_cyclical: bool = False) -> List[str]:
    """
    Get list of date feature names.
    
    Args:
        include_cyclical: Whether to include cyclical features
        
    Returns:
        List of feature names
    """
    features = [
        'day',
        'day_of_week',
        'week_of_year',
        'month',
        'quarter',
        'year',
        'is_weekend'
    ]
    
    if include_cyclical:
        features.extend([
            'day_of_week_sin',
            'day_of_week_cos',
            'day_sin',
            'day_cos',
            'month_sin',
            'month_cos'
        ])
    
    return features
