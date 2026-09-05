"""
Feature Creation Script

This script creates all features for the preprocessed data.
Run this after data preprocessing is complete.

Usage:
    python scripts/create_features.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import load_processed_data
from src.features.feature_engineering import prepare_features_for_training
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run the feature engineering pipeline."""
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING SCRIPT")
    logger.info("=" * 70)
    
    # Check if processed data exists
    processed_path = Path("data/processed")
    if not (processed_path / "train.parquet").exists():
        logger.error("✗ Processed data not found!")
        logger.info("Please run: python scripts/prepare_data.py")
        return
    
    # Load processed data
    logger.info("\nLoading processed data...")
    train_df, val_df, test_df = load_processed_data("data/processed")
    
    # Create features
    logger.info("\nCreating features...")
    train_features, val_features, test_features = prepare_features_for_training(
        train_df,
        val_df,
        test_df,
        output_dir="data/processed"
    )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Training features:   {train_features.shape}")
    logger.info(f"Validation features: {val_features.shape}")
    logger.info(f"Test features:       {test_features.shape}")
    logger.info(f"\nTotal features: {train_features.shape[1]}")
    logger.info("\nFeature files saved to: data/processed/")
    logger.info("  - train_features.parquet")
    logger.info("  - val_features.parquet")
    logger.info("  - test_features.parquet")
    logger.info("\n✓ Feature engineering complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
