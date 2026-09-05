"""
Lag Features Module

This module creates lag features (past values) for time series forecasting.
Lag features allow models to use historical sales to predict future sales.

Key considerations:
- Lag features must be created within each time series (store-item combination)
- Must prevent data leakage by not using future information
- Missing values occur at the start of each series (handled appropriately)

Example:
    If we want to predict sales on day 100:
    - lag_1: sales on day 99
    - lag_7: sales on day 93 (one week ago)
    - lag_28: sales on day 72 (four weeks ago)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LagFeatureEngineer:
    """
    Engineer lag features for time series forecasting.
    
    Lag features are historical values of the target variable.
    They are crucial for capturing autocorrelation patterns.
    """
    
    def __init__(self, lag_periods: Optional[List[int]] = None):
        """
        Initialize the lag feature engineer.
        
        Args:
            lag_periods: List of lag periods (in days). Default: [1, 7, 14, 28]
        """
        self.lag_periods = lag_periods or [1, 7, 14, 28]
        logger.info(f"Lag periods: {self.lag_periods}")
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales',
        group_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for a time series.
        
        Args:
            df: Input DataFrame (must be sorted by date within each group)
            target_column: Column to create lags from (typically 'sales')
            group_columns: Columns defining separate time series (e.g., ['store_id', 'item_id'])
                          If None, treats entire dataset as one series
        
        Returns:
            DataFrame with additional lag feature columns
            
        Important:
            - Input DataFrame must be sorted by date within each group
            - Lag features will have NaN values at the start of each series
            - These NaN values should be handled appropriately (drop or fill)
            
        Example:
            >>> df = create_lag_features(df, target_column='sales', 
            ...                          group_columns=['store_id', 'item_id'])
            >>> # Creates: lag_1, lag_7, lag_14, lag_28
        """
        logger.info(f"Creating lag features for '{target_column}'...")
        
        df = df.copy()
        
        # Default group columns for M5 dataset
        if group_columns is None:
            group_columns = ['store_id', 'item_id']
        
        # Create lag features
        for lag in self.lag_periods:
            feature_name = f'lag_{lag}'
            
            # Create lag within each group (store-item combination)
            df[feature_name] = df.groupby(group_columns)[target_column].shift(lag)
            
            # Track how many NaN values were created
            nan_count = df[feature_name].isna().sum()
            logger.info(f"  Created {feature_name} ({nan_count:,} NaN values)")
        
        logger.info(f"✓ Created {len(self.lag_periods)} lag features")
        
        return df
    
    def get_lag_feature_names(self) -> List[str]:
        """
        Get list of lag feature names.
        
        Returns:
            List of feature names
        """
        return [f'lag_{lag}' for lag in self.lag_periods]


def create_diff_features(
    df: pd.DataFrame,
    target_column: str = 'sales',
    group_columns: Optional[List[str]] = None,
    periods: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create difference features (change from previous period).
    
    Difference features capture the change in sales rather than absolute level:
    - diff_1: sales today - sales yesterday
    - diff_7: sales today - sales last week
    
    These can be useful for non-stationary time series.
    
    Args:
        df: Input DataFrame
        target_column: Column to create differences from
        group_columns: Columns defining separate time series
        periods: List of periods for differencing. Default: [1, 7]
        
    Returns:
        DataFrame with additional difference features
    """
    logger.info(f"Creating difference features for '{target_column}'...")
    
    df = df.copy()
    
    if group_columns is None:
        group_columns = ['store_id', 'item_id']
    
    if periods is None:
        periods = [1, 7]
    
    for period in periods:
        feature_name = f'diff_{period}'
        
        # Calculate difference within each group
        df[feature_name] = df.groupby(group_columns)[target_column].diff(period)
        
        nan_count = df[feature_name].isna().sum()
        logger.info(f"  Created {feature_name} ({nan_count:,} NaN values)")
    
    logger.info(f"✓ Created {len(periods)} difference features")
    
    return df


def handle_lag_missing_values(
    df: pd.DataFrame,
    lag_features: List[str],
    method: str = 'drop'
) -> pd.DataFrame:
    """
    Handle missing values in lag features.
    
    Strategies:
    - 'drop': Remove rows with any missing lag features (safest for training)
    - 'fill_zero': Fill with 0 (assumes no sales)
    - 'fill_forward': Forward fill within each group
    - 'fill_median': Fill with median of that feature
    
    Args:
        df: DataFrame with lag features
        lag_features: List of lag feature names
        method: Method to handle missing values
        
    Returns:
        DataFrame with missing values handled
        
    Note:
        For training data, 'drop' is recommended to avoid introducing bias.
        For prediction, forward filling or median imputation may be needed.
    """
    logger.info(f"Handling missing values in lag features (method: {method})...")
    
    df = df.copy()
    
    initial_rows = len(df)
    
    if method == 'drop':
        df = df.dropna(subset=lag_features)
        dropped_rows = initial_rows - len(df)
        logger.info(f"  Dropped {dropped_rows:,} rows with missing lag values")
    
    elif method == 'fill_zero':
        df[lag_features] = df[lag_features].fillna(0)
        logger.info("  Filled missing values with 0")
    
    elif method == 'fill_forward':
        # This should be done within groups
        logger.warning("Forward fill not implemented for grouped data yet")
        df[lag_features] = df[lag_features].fillna(method='ffill')
    
    elif method == 'fill_median':
        for feature in lag_features:
            median_val = df[feature].median()
            df[feature] = df[feature].fillna(median_val)
        logger.info("  Filled missing values with median")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info("✓ Missing values handled")
    
    return df
