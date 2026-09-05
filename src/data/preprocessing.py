"""
Data Preprocessing Module

This module transforms the M5 dataset from wide format to long format,
merges datasets, handles missing values, and prepares data for feature engineering.

Key transformations:
- Wide-to-long format conversion for sales data
- Merge sales, calendar, and price data
- Handle missing prices
- Create time-series friendly format
- Split data chronologically (train/validation/test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class M5Preprocessor:
    """
    Preprocessor for M5 Forecasting dataset.
    
    This class handles the transformation of raw M5 data into a format
    suitable for time-series forecasting models.
    
    The main challenge with M5 data is that sales are in wide format
    (one column per day), which needs to be converted to long format
    (one row per product-store-day combination).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary. If None, loads from default config file.
        """
        self.config = config or load_config()
        self.data_config = self.config.get('data', {})
    
    def melt_sales_data(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert sales data from wide format to long format.
        
        The original sales data has columns like:
        id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ..., d_1913
        
        This converts it to:
        id, item_id, dept_id, cat_id, store_id, state_id, d, sales
        
        Args:
            sales_df: Sales DataFrame in wide format
            
        Returns:
            Sales DataFrame in long format
            
        Note:
            This operation can be memory-intensive for the full dataset.
            The resulting DataFrame will have ~60 million rows.
        """
        logger.info("Converting sales data from wide to long format...")
        logger.info(f"Input shape: {sales_df.shape}")
        
        # Identify columns
        id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        day_columns = [col for col in sales_df.columns if col.startswith('d_')]
        
        logger.info(f"Found {len(day_columns)} daily sales columns")
        
        # Melt the DataFrame
        sales_long = pd.melt(
            sales_df,
            id_vars=id_columns,
            value_vars=day_columns,
            var_name='d',
            value_name='sales'
        )
        
        logger.info(f"Output shape: {sales_long.shape}")
        logger.info("✓ Conversion complete")
        
        return sales_long
    
    def merge_datasets(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge sales, calendar, and price data.
        
        This creates a unified dataset with:
        - Sales information (item, store, quantity sold)
        - Calendar information (date, holidays, events)
        - Price information (sell price per week)
        
        Args:
            sales_df: Sales DataFrame (long format)
            calendar_df: Calendar DataFrame
            prices_df: Prices DataFrame
            
        Returns:
            Merged DataFrame
            
        Note:
            Prices are provided at weekly granularity (wm_yr_wk),
            while sales are daily. We merge and forward-fill prices.
        """
        logger.info("Merging datasets...")
        
        # Merge sales with calendar
        logger.info("Merging sales with calendar data...")
        df = sales_df.merge(calendar_df, on='d', how='left')
        logger.info(f"After calendar merge: {df.shape}")
        
        # Merge with prices
        logger.info("Merging with price data...")
        df = df.merge(
            prices_df,
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )
        logger.info(f"After price merge: {df.shape}")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Check for missing prices
        missing_prices = df['sell_price'].isna().sum()
        if missing_prices > 0:
            logger.warning(f"Missing prices: {missing_prices} rows ({missing_prices / len(df) * 100:.2f}%)")
            logger.info("Missing prices will be handled in preprocessing")
        
        logger.info("✓ Datasets merged successfully")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the merged dataset.
        
        Strategy:
        - Missing prices: Forward fill within same item-store, then backward fill
        - Event columns: Fill with 'None' or 0 (no event)
        
        Args:
            df: Merged DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        # Sort by store, item, and date for proper forward filling
        df = df.sort_values(['store_id', 'item_id', 'date']).copy()
        
        # Handle missing prices
        missing_before = df['sell_price'].isna().sum()
        
        if missing_before > 0:
            logger.info(f"Filling {missing_before} missing prices...")
            
            # Forward fill prices within each store-item group
            df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].fillna(method='ffill')
            
            # Backward fill for remaining missing values (at the start of series)
            df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].fillna(method='bfill')
            
            missing_after = df['sell_price'].isna().sum()
            logger.info(f"Missing prices after filling: {missing_after}")
            
            # If still have missing prices, fill with median price of the item
            if missing_after > 0:
                logger.warning("Some prices still missing, filling with item median")
                item_median_price = df.groupby('item_id')['sell_price'].transform('median')
                df['sell_price'] = df['sell_price'].fillna(item_median_price)
        
        # Handle missing event columns (if they exist)
        event_cols = [col for col in df.columns if 'event' in col.lower()]
        for col in event_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('None')
        
        # Check snap columns
        snap_cols = [col for col in df.columns if col.startswith('snap_')]
        for col in snap_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        logger.info("✓ Missing values handled")
        
        return df
    
    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        train_end_date: Optional[str] = None,
        val_end_date: Optional[str] = None,
        test_end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test splits.
        
        This ensures NO data leakage - training data is always before validation,
        and validation is always before test data.
        
        Args:
            df: Preprocessed DataFrame
            train_end_date: Last date of training period (YYYY-MM-DD)
            val_end_date: Last date of validation period (YYYY-MM-DD)
            test_end_date: Last date of test period (YYYY-MM-DD)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Note:
            Default dates from config:
            - Train: up to 2016-03-27
            - Validation: 2016-03-28 to 2016-04-24 (28 days)
            - Test: 2016-04-25 to 2016-05-22 (28 days)
        """
        logger.info("Creating train/validation/test splits...")
        
        # Use dates from config if not provided
        if train_end_date is None:
            train_end_date = self.data_config.get('train_end_date', '2016-03-27')
        if val_end_date is None:
            val_end_date = self.data_config.get('validation_end_date', '2016-04-24')
        if test_end_date is None:
            test_end_date = self.data_config.get('test_end_date', '2016-05-22')
        
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        test_end = pd.to_datetime(test_end_date)
        
        # Create splits
        train_df = df[df['date'] <= train_end].copy()
        val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
        test_df = df[(df['date'] > val_end) & (df['date'] <= test_end)].copy()
        
        logger.info(f"Train set: {train_df.shape} | Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"Val set:   {val_df.shape} | Date range: {val_df['date'].min()} to {val_df['date'].max()}")
        logger.info(f"Test set:  {test_df.shape} | Date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Verify no overlap
        assert train_df['date'].max() < val_df['date'].min(), "Train and validation overlap!"
        assert val_df['date'].max() < test_df['date'].min(), "Validation and test overlap!"
        
        logger.info("✓ Chronological splits created (no data leakage)")
        
        return train_df, val_df, test_df
    
    def preprocess_full_pipeline(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.
        
        Steps:
        1. Convert sales data to long format
        2. Merge all datasets
        3. Handle missing values
        4. Create train/val/test splits
        5. Save processed data
        
        Args:
            sales_df: Raw sales DataFrame
            calendar_df: Raw calendar DataFrame
            prices_df: Raw prices DataFrame
            output_dir: Directory to save processed data (optional)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Melt sales data
        sales_long = self.melt_sales_data(sales_df)
        
        # Step 2: Merge datasets
        merged_df = self.merge_datasets(sales_long, calendar_df, prices_df)
        
        # Step 3: Handle missing values
        processed_df = self.handle_missing_values(merged_df)
        
        # Step 4: Create splits
        train_df, val_df, test_df = self.create_train_val_test_split(processed_df)
        
        # Step 5: Save processed data (if output directory provided)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving processed data to {output_dir}...")
            
            # Save as parquet for efficiency
            train_df.to_parquet(output_path / "train.parquet", index=False)
            val_df.to_parquet(output_path / "val.parquet", index=False)
            test_df.to_parquet(output_path / "test.parquet", index=False)
            
            # Also save the full merged dataset
            processed_df.to_parquet(output_path / "processed_full.parquet", index=False)
            
            logger.info("✓ Processed data saved")
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return train_df, val_df, test_df


def load_processed_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train/val/test data.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)
    
    logger.info("Loading processed data...")
    
    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")
    
    logger.info(f"✓ Train set: {train_df.shape}")
    logger.info(f"✓ Val set: {val_df.shape}")
    logger.info(f"✓ Test set: {test_df.shape}")
    
    return train_df, val_df, test_df
