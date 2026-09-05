"""
Data Validation Module

This module provides schema validation and data quality checks for the M5 dataset.
It ensures data integrity before processing and helps catch issues early.

Key validations:
- Schema validation (correct columns, data types)
- Missing value detection
- Duplicate detection
- Date range validation
- Value range validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class M5DataValidator:
    """
    Validator for M5 Forecasting dataset.
    
    This class performs comprehensive validation checks on the M5 dataset
    to ensure data quality and prevent issues in downstream processing.
    """
    
    def __init__(self):
        """Initialize the validator with schema definitions."""
        self.sales_schema = self._create_sales_schema()
        self.calendar_schema = self._create_calendar_schema()
        self.prices_schema = self._create_prices_schema()
    
    def _create_sales_schema(self) -> DataFrameSchema:
        """
        Create pandera schema for sales data validation.
        
        The sales data contains:
        - Item identifiers (id, item_id, dept_id, cat_id, store_id, state_id)
        - Daily sales columns (d_1, d_2, ..., d_1913)
        
        Returns:
            Pandera DataFrameSchema for sales data
        """
        # Note: We'll validate core columns; daily sales columns are validated separately
        return DataFrameSchema(
            {
                "id": Column(str, nullable=False),
                "item_id": Column(str, nullable=False),
                "dept_id": Column(str, nullable=False),
                "cat_id": Column(str, nullable=False),
                "store_id": Column(str, nullable=False),
                "state_id": Column(str, nullable=False)
            },
            strict=False  # Allow additional columns (d_1, d_2, etc.)
        )
    
    def _create_calendar_schema(self) -> DataFrameSchema:
        """
        Create pandera schema for calendar data validation.
        
        Returns:
            Pandera DataFrameSchema for calendar data
        """
        return DataFrameSchema(
            {
                "date": Column(str, nullable=False),
                "wm_yr_wk": Column(pa.Int, nullable=False),
                "weekday": Column(str, nullable=False),
                "wday": Column(pa.Int, Check.in_range(1, 7), nullable=False),
                "month": Column(pa.Int, Check.in_range(1, 12), nullable=False),
                "year": Column(pa.Int, nullable=False),
                "d": Column(str, nullable=False),
            },
            strict=False  # Allow event columns
        )
    
    def _create_prices_schema(self) -> DataFrameSchema:
        """
        Create pandera schema for price data validation.
        
        Returns:
            Pandera DataFrameSchema for price data
        """
        return DataFrameSchema(
            {
                "store_id": Column(str, nullable=False),
                "item_id": Column(str, nullable=False),
                "wm_yr_wk": Column(pa.Int, nullable=False),
                "sell_price": Column(pa.Float, Check.greater_than(0), nullable=False)
            }
        )
    
    def validate_sales_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate sales data.
        
        Args:
            df: Sales DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Schema validation
            self.sales_schema.validate(df, lazy=True)
            logger.info("✓ Sales data schema validation passed")
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Schema validation issues found: {e}")
            issues.append(f"Schema errors: {e}")
        
        # Check for daily sales columns
        day_columns = [col for col in df.columns if col.startswith('d_')]
        if len(day_columns) == 0:
            issues.append("No daily sales columns (d_1, d_2, ...) found")
        else:
            logger.info(f"✓ Found {len(day_columns)} daily sales columns")
        
        # Check for negative sales
        for col in day_columns:
            if (df[col] < 0).any():
                issues.append(f"Negative sales values found in {col}")
                break
        
        # Check for duplicates
        id_cols = ['id']
        duplicates = df.duplicated(subset=id_cols).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows based on 'id' column")
        else:
            logger.info("✓ No duplicate rows found")
        
        # Check for missing values in key columns
        key_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        for col in key_columns:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"Missing values in {col}: {missing}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_calendar_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate calendar data.
        
        Args:
            df: Calendar DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Schema validation
            self.calendar_schema.validate(df, lazy=True)
            logger.info("✓ Calendar data schema validation passed")
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Schema validation issues found: {e}")
            issues.append(f"Schema errors: {e}")
        
        # Check date format and range
        try:
            df['date_parsed'] = pd.to_datetime(df['date'])
            date_range = df['date_parsed'].max() - df['date_parsed'].min()
            logger.info(f"✓ Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()} ({date_range.days} days)")
        except Exception as e:
            issues.append(f"Date parsing error: {e}")
        
        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.warning(f"Missing values in {col}: {missing}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['date']).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate dates")
        else:
            logger.info("✓ No duplicate dates found")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_prices_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data.
        
        Args:
            df: Prices DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Schema validation
            self.prices_schema.validate(df, lazy=True)
            logger.info("✓ Prices data schema validation passed")
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Schema validation issues found: {e}")
            issues.append(f"Schema errors: {e}")
        
        # Check for missing values
        missing = df.isna().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            issues.append(f"Missing values in price data")
        else:
            logger.info("✓ No missing values in price data")
        
        # Check price distribution
        price_stats = df['sell_price'].describe()
        logger.info(f"Price statistics:\n{price_stats}")
        
        # Check for unrealistic prices
        if (df['sell_price'] > 1000).any():
            logger.warning("Some prices exceed $1000 - verify if this is expected")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_all(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all M5 dataset files.
        
        Args:
            sales_df: Sales DataFrame
            calendar_df: Calendar DataFrame
            prices_df: Prices DataFrame
            
        Returns:
            Dictionary with validation results for each dataset
        """
        logger.info("=" * 60)
        logger.info("VALIDATING M5 DATASET")
        logger.info("=" * 60)
        
        results = {}
        
        logger.info("\n1. Validating Sales Data...")
        results['sales'] = self.validate_sales_data(sales_df)
        
        logger.info("\n2. Validating Calendar Data...")
        results['calendar'] = self.validate_calendar_data(calendar_df)
        
        logger.info("\n3. Validating Prices Data...")
        results['prices'] = self.validate_prices_data(prices_df)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        all_valid = True
        for dataset_name, (is_valid, issues) in results.items():
            status = "✓ PASSED" if is_valid else "✗ FAILED"
            logger.info(f"{dataset_name.upper()}: {status}")
            if issues:
                for issue in issues:
                    logger.warning(f"  - {issue}")
            all_valid = all_valid and is_valid
        
        if all_valid:
            logger.info("\n✓ All validations passed!")
        else:
            logger.warning("\n✗ Some validations failed. Review issues above.")
        
        return results


def load_and_validate_data(data_dir: str = "data/raw") -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load and validate M5 dataset files.
    
    This is a convenience function that loads all three M5 dataset files
    and runs validation checks on them.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        Tuple of (sales_df, calendar_df, prices_df) if validation passes,
        None if validation fails
    """
    data_path = Path(data_dir)
    
    try:
        # Load data
        logger.info("Loading M5 dataset files...")
        sales_df = pd.read_csv(data_path / "sales_train_evaluation.csv")
        calendar_df = pd.read_csv(data_path / "calendar.csv")
        prices_df = pd.read_csv(data_path / "sell_prices.csv")
        
        logger.info(f"✓ Sales data: {sales_df.shape}")
        logger.info(f"✓ Calendar data: {calendar_df.shape}")
        logger.info(f"✓ Prices data: {prices_df.shape}")
        
        # Validate
        validator = M5DataValidator()
        results = validator.validate_all(sales_df, calendar_df, prices_df)
        
        # Check if all validations passed
        all_valid = all(is_valid for is_valid, _ in results.values())
        
        if all_valid:
            return sales_df, calendar_df, prices_df
        else:
            logger.error("Validation failed. Please fix data issues before proceeding.")
            return None
            
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        logger.info("Please download the M5 dataset first. Run: python src/data/download_data.py")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
