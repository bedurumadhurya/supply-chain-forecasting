"""
Data Preparation Script

This script runs the complete data pipeline:
1. Check if M5 dataset exists
2. Load and validate raw data
3. Preprocess data (melt, merge, clean)
4. Create train/val/test splits
5. Save processed data

Usage:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.download_data import check_dataset_exists, print_download_instructions
from src.data.validation import load_and_validate_data
from src.data.preprocessing import M5Preprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run the complete data preparation pipeline."""
    logger.info("=" * 70)
    logger.info("M5 DATASET PREPARATION PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Check if dataset exists
    logger.info("\nStep 1: Checking for M5 dataset files...")
    file_status = check_dataset_exists()
    
    all_exist = all(file_status.values())
    
    if not all_exist:
        logger.error("✗ M5 dataset files not found!")
        print_download_instructions()
        return
    
    logger.info("✓ All M5 dataset files found")
    
    # Step 2: Load and validate data
    logger.info("\nStep 2: Loading and validating data...")
    result = load_and_validate_data()
    
    if result is None:
        logger.error("✗ Data validation failed!")
        return
    
    sales_df, calendar_df, prices_df = result
    logger.info("✓ Data loaded and validated")
    
    # Step 3: Preprocess data
    logger.info("\nStep 3: Preprocessing data...")
    logger.info("Note: This may take several minutes for the full dataset...")
    
    preprocessor = M5Preprocessor()
    
    try:
        train_df, val_df, test_df = preprocessor.preprocess_full_pipeline(
            sales_df,
            calendar_df,
            prices_df,
            output_dir="data/processed"
        )
        
        logger.info("✓ Data preprocessing complete!")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("PREPARATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Training set:   {len(train_df):,} rows")
        logger.info(f"Validation set: {len(val_df):,} rows")
        logger.info(f"Test set:       {len(test_df):,} rows")
        logger.info(f"Total:          {len(train_df) + len(val_df) + len(test_df):,} rows")
        logger.info("\nProcessed data saved to: data/processed/")
        logger.info("Files:")
        logger.info("  - train.parquet")
        logger.info("  - val.parquet")
        logger.info("  - test.parquet")
        logger.info("  - processed_full.parquet")
        logger.info("\n✓ Data preparation pipeline complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
