"""
Rolling Features Module

This module creates rolling window statistics (moving averages, standard deviations, etc.)
These features capture recent trends and volatility in the time series.

Key features:
- Rolling mean (moving average): Captures recent trend
- Rolling std (moving standard deviation): Captures volatility
- Rolling min/max: Captures range of recent values

Important:
- Rolling windows must be calculated within each time series group
- Must prevent data leakage by only using past data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RollingFeatureEngineer:
    """
    Engineer rolling window features for time series forecasting.
    
    Rolling features summarize recent history and help models understand:
    - Recent trends (moving averages)
    - Recent volatility (moving standard deviations)
    - Recent extremes (min/max)
    """
    
    def __init__(self, windows: Optional[List[Dict]] = None):
        """
        Initialize the rolling feature engineer.
        
        Args:
            windows: List of window configurations. Each dict should have:
                    - 'window': Window size in days
                    - 'functions': List of functions to apply ('mean', 'std', 'min', 'max')
                    
                    Default: [
                        {'window': 7, 'functions': ['mean', 'std']},
                        {'window': 14, 'functions': ['mean', 'std']},
                        {'window': 28, 'functions': ['mean', 'std', 'min', 'max']}
                    ]
        """
        if windows is None:
            self.windows = [
                {'window': 7, 'functions': ['mean', 'std']},
                {'window': 14, 'functions': ['mean', 'std']},
                {'window': 28, 'functions': ['mean', 'std', 'min', 'max']}
            ]
        else:
            self.windows = windows
        
        logger.info(f"Rolling windows configured: {len(self.windows)} window sizes")
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales',
        group_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame (must be sorted by date within each group)
            target_column: Column to calculate rolling stats from
            group_columns: Columns defining separate time series
            
        Returns:
            DataFrame with additional rolling feature columns
            
        Important:
            - Input must be sorted by date within each group
            - Rolling calculations use only past data (no future leakage)
            - First N rows of each series will have NaN (where N = window size)
            
        Example:
            >>> df = create_rolling_features(df, target_column='sales',
            ...                              group_columns=['store_id', 'item_id'])
            >>> # Creates: rolling_mean_7, rolling_std_7, rolling_mean_14, etc.
        """
        logger.info(f"Creating rolling features for '{target_column}'...")
        
        df = df.copy()
        
        if group_columns is None:
            group_columns = ['store_id', 'item_id']
        
        feature_count = 0
        
        # Create rolling features for each window configuration
        for window_config in self.windows:
            window_size = window_config['window']
            functions = window_config['functions']
            
            logger.info(f"  Processing window={window_size}, functions={functions}")
            
            # Group by store-item to calculate rolling stats separately for each series
            grouped = df.groupby(group_columns)[target_column]
            
            for func_name in functions:
                feature_name = f'rolling_{func_name}_{window_size}'
                
                # Calculate rolling statistic
                if func_name == 'mean':
                    df[feature_name] = grouped.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).mean()
                    )
                elif func_name == 'std':
                    df[feature_name] = grouped.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).std()
                    )
                elif func_name == 'min':
                    df[feature_name] = grouped.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).min()
                    )
                elif func_name == 'max':
                    df[feature_name] = grouped.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).max()
                    )
                elif func_name == 'sum':
                    df[feature_name] = grouped.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).sum()
                    )
                else:
                    logger.warning(f"Unknown function: {func_name}, skipping")
                    continue
                
                feature_count += 1
        
        logger.info(f"✓ Created {feature_count} rolling features")
        
        return df
    
    def create_expanding_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales',
        group_columns: Optional[List[str]] = None,
        functions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create expanding window features (cumulative statistics).
        
        Expanding window calculates statistics from the start of the series
        up to the current point. This captures the overall trend.
        
        Args:
            df: Input DataFrame
            target_column: Column to calculate stats from
            group_columns: Columns defining separate time series
            functions: List of functions ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            DataFrame with expanding window features
            
        Note:
            These features grow over time and can capture long-term trends.
        """
        logger.info(f"Creating expanding window features for '{target_column}'...")
        
        df = df.copy()
        
        if group_columns is None:
            group_columns = ['store_id', 'item_id']
        
        if functions is None:
            functions = ['mean', 'std']
        
        grouped = df.groupby(group_columns)[target_column]
        
        for func_name in functions:
            feature_name = f'expanding_{func_name}'
            
            if func_name == 'mean':
                df[feature_name] = grouped.transform(lambda x: x.expanding(min_periods=1).mean())
            elif func_name == 'std':
                df[feature_name] = grouped.transform(lambda x: x.expanding(min_periods=1).std())
            elif func_name == 'min':
                df[feature_name] = grouped.transform(lambda x: x.expanding(min_periods=1).min())
            elif func_name == 'max':
                df[feature_name] = grouped.transform(lambda x: x.expanding(min_periods=1).max())
            elif func_name == 'sum':
                df[feature_name] = grouped.transform(lambda x: x.expanding(min_periods=1).sum())
        
        logger.info(f"✓ Created {len(functions)} expanding window features")
        
        return df
    
    def get_rolling_feature_names(self) -> List[str]:
        """
        Get list of rolling feature names.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        for window_config in self.windows:
            window_size = window_config['window']
            functions = window_config['functions']
            
            for func_name in functions:
                feature_names.append(f'rolling_{func_name}_{window_size}')
        
        return feature_names


def create_rolling_ratio_features(
    df: pd.DataFrame,
    target_column: str = 'sales',
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create ratio features between different rolling windows.
    
    These ratios can indicate acceleration or deceleration in sales:
    - ratio_7_28: Recent 7-day average / 28-day average
    - If > 1: Recent sales higher than longer-term average
    - If < 1: Recent sales lower than longer-term average
    
    Args:
        df: DataFrame with existing rolling features
        target_column: Target column name
        group_columns: Grouping columns
        
    Returns:
        DataFrame with ratio features
    """
    logger.info("Creating rolling ratio features...")
    
    df = df.copy()
    
    if group_columns is None:
        group_columns = ['store_id', 'item_id']
    
    # Check if rolling features exist
    if 'rolling_mean_7' in df.columns and 'rolling_mean_28' in df.columns:
        # Ratio of 7-day to 28-day average
        df['ratio_7_28'] = df['rolling_mean_7'] / (df['rolling_mean_28'] + 1e-8)  # Add small epsilon to avoid division by zero
        logger.info("  Created ratio_7_28")
    
    if 'rolling_mean_14' in df.columns and 'rolling_mean_28' in df.columns:
        # Ratio of 14-day to 28-day average
        df['ratio_14_28'] = df['rolling_mean_14'] / (df['rolling_mean_28'] + 1e-8)
        logger.info("  Created ratio_14_28")
    
    logger.info("✓ Created rolling ratio features")
    
    return df
