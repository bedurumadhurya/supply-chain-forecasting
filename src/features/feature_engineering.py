"""
Feature Engineering Pipeline

This module orchestrates the complete feature engineering process:
1. Date features
2. Lag features
3. Rolling features
4. Price features
5. Category features

All features are created with careful attention to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .date_features import DateFeatureEngineer, add_holiday_features, get_date_feature_names
from .lag_features import LagFeatureEngineer, create_diff_features, handle_lag_missing_values
from .rolling_features import RollingFeatureEngineer, create_rolling_ratio_features
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline for demand forecasting.
    
    This pipeline creates all features needed for forecasting models
    while ensuring no data leakage occurs.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            config: Configuration dictionary. If None, loads from default config file.
        """
        self.config = config or load_config()
        self.feature_config = self.config.get('features', {})
        
        # Initialize feature engineers
        lag_periods = self.feature_config.get('lag_periods', [1, 7, 14, 28])
        rolling_windows = self.feature_config.get('rolling_windows', [
            {'window': 7, 'functions': ['mean', 'std']},
            {'window': 14, 'functions': ['mean', 'std']},
            {'window': 28, 'functions': ['mean', 'std', 'min', 'max']}
        ])
        
        self.date_engineer = DateFeatureEngineer()
        self.lag_engineer = LagFeatureEngineer(lag_periods=lag_periods)
        self.rolling_engineer = RollingFeatureEngineer(windows=rolling_windows)
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_cyclical: bool = False,
        include_diff: bool = False
    ) -> pd.DataFrame:
        """
        Create all features in the correct order.
        
        Order is important:
        1. Date features (no dependencies)
        2. Lag features (require sorted data)
        3. Rolling features (require sorted data)
        4. Derived features (ratios, etc.)
        
        Args:
            df: Input DataFrame (preprocessed data)
            include_cyclical: Whether to include cyclical date encoding
            include_diff: Whether to include difference features
            
        Returns:
            DataFrame with all engineered features
            
        Important:
            - Input must be sorted by store_id, item_id, date
            - Lag and rolling features will create NaN values at series starts
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        df = df.copy()
        initial_shape = df.shape
        logger.info(f"Input shape: {initial_shape}")
        
        # Ensure data is sorted properly
        logger.info("\nSorting data by store_id, item_id, date...")
        df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
        
        # Step 1: Date features
        logger.info("\n1. Creating date features...")
        df = self.date_engineer.create_date_features(df, date_column='date')
        
        if include_cyclical:
            df = self.date_engineer.create_cyclical_features(df)
        
        # Step 2: Holiday/Event features
        logger.info("\n2. Processing holiday and event features...")
        df = add_holiday_features(df)
        
        # Step 3: Price features
        logger.info("\n3. Creating price features...")
        df = self._create_price_features(df)
        
        # Step 4: Lag features
        logger.info("\n4. Creating lag features...")
        df = self.lag_engineer.create_lag_features(
            df,
            target_column='sales',
            group_columns=['store_id', 'item_id']
        )
        
        if include_diff:
            df = create_diff_features(
                df,
                target_column='sales',
                group_columns=['store_id', 'item_id']
            )
        
        # Step 5: Rolling features
        logger.info("\n5. Creating rolling features...")
        df = self.rolling_engineer.create_rolling_features(
            df,
            target_column='sales',
            group_columns=['store_id', 'item_id']
        )
        
        # Step 6: Rolling ratio features
        logger.info("\n6. Creating rolling ratio features...")
        df = create_rolling_ratio_features(
            df,
            target_column='sales',
            group_columns=['store_id', 'item_id']
        )
        
        # Step 7: Categorical encoding
        logger.info("\n7. Encoding categorical features...")
        df = self._encode_categorical_features(df)
        
        final_shape = df.shape
        logger.info(f"\nFinal shape: {final_shape}")
        logger.info(f"Added {final_shape[1] - initial_shape[1]} features")
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-related features.
        
        Features:
        - price_momentum: Price change over time
        - price_relative_to_mean: Current price / average price
        
        Args:
            df: DataFrame with sell_price column
            
        Returns:
            DataFrame with price features
        """
        if 'sell_price' not in df.columns:
            logger.warning("sell_price column not found, skipping price features")
            return df
        
        # Price change (7-day)
        df['price_change_7'] = df.groupby(['store_id', 'item_id'])['sell_price'].diff(7)
        
        # Price relative to item average
        df['price_relative'] = df.groupby('item_id')['sell_price'].transform(
            lambda x: x / (x.mean() + 1e-8)
        )
        
        # Price rank within item (higher price = higher rank)
        df['price_rank'] = df.groupby('item_id')['sell_price'].rank(pct=True)
        
        logger.info("  ✓ Created 3 price features")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for modeling.
        
        Uses label encoding for tree-based models.
        For neural networks, use one-hot encoding or embeddings separately.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded categories
        """
        categorical_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        for col in categorical_columns:
            if col in df.columns:
                # Create label encoding
                df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        
        logger.info(f"  ✓ Encoded {len(categorical_columns)} categorical features")
        
        return df
    
    def get_feature_columns(
        self,
        include_target: bool = False,
        include_identifiers: bool = False
    ) -> List[str]:
        """
        Get list of feature column names.
        
        Args:
            include_target: Whether to include 'sales' column
            include_identifiers: Whether to include id columns
            
        Returns:
            List of feature column names
        """
        # Core features
        features = []
        
        # Date features
        features.extend(get_date_feature_names(include_cyclical=False))
        
        # Lag features
        features.extend(self.lag_engineer.get_lag_feature_names())
        
        # Rolling features
        features.extend(self.rolling_engineer.get_rolling_feature_names())
        
        # Price features
        features.extend(['price_change_7', 'price_relative', 'price_rank', 'sell_price'])
        
        # Rolling ratios
        features.extend(['ratio_7_28', 'ratio_14_28'])
        
        # Categorical encoded features
        features.extend([
            'item_id_encoded', 'dept_id_encoded', 'cat_id_encoded',
            'store_id_encoded', 'state_id_encoded'
        ])
        
        # Event features
        features.extend(['has_event'])
        
        # SNAP features
        features.extend(['snap_CA', 'snap_TX', 'snap_WI'])
        
        if include_target:
            features.append('sales')
        
        if include_identifiers:
            features.extend(['id', 'item_id', 'store_id', 'date'])
        
        return features


def prepare_features_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for all splits (train/val/test).
    
    This function:
    1. Creates features for all splits
    2. Handles missing values in lag features
    3. Saves feature-engineered data
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Directory to save feature-engineered data
        
    Returns:
        Tuple of (train_features, val_features, test_features)
        
    Important:
        Features are created separately for each split to prevent leakage,
        but using the same feature engineering logic.
    """
    logger.info("=" * 70)
    logger.info("PREPARING FEATURES FOR TRAINING")
    logger.info("=" * 70)
    
    pipeline = FeaturePipeline()
    
    # Create features for each split
    logger.info("\nProcessing training set...")
    train_features = pipeline.create_all_features(train_df)
    
    logger.info("\nProcessing validation set...")
    val_features = pipeline.create_all_features(val_df)
    
    logger.info("\nProcessing test set...")
    test_features = pipeline.create_all_features(test_df)
    
    # Handle missing values in lag features (drop for training)
    logger.info("\nHandling missing values in lag features...")
    
    lag_features = pipeline.lag_engineer.get_lag_feature_names()
    rolling_features = pipeline.rolling_engineer.get_rolling_feature_names()
    feature_list = lag_features + rolling_features
    
    # For training: drop rows with missing lag/rolling features
    train_features = handle_lag_missing_values(train_features, feature_list, method='drop')
    
    # For validation/test: forward fill or drop (depending on use case)
    # Here we'll drop to maintain consistency
    val_features = handle_lag_missing_values(val_features, feature_list, method='drop')
    test_features = handle_lag_missing_values(test_features, feature_list, method='drop')
    
    logger.info(f"\nFinal shapes:")
    logger.info(f"  Train: {train_features.shape}")
    logger.info(f"  Val:   {val_features.shape}")
    logger.info(f"  Test:  {test_features.shape}")
    
    # Save feature-engineered data
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving feature-engineered data to {output_dir}...")
        train_features.to_parquet(output_path / "train_features.parquet", index=False)
        val_features.to_parquet(output_path / "val_features.parquet", index=False)
        test_features.to_parquet(output_path / "test_features.parquet", index=False)
        logger.info("✓ Feature data saved")
    
    logger.info("=" * 70)
    logger.info("FEATURE PREPARATION COMPLETE")
    logger.info("=" * 70)
    
    return train_features, val_features, test_features
