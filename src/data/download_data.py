"""
M5 Dataset Download Module

This module provides utilities for downloading the M5 Forecasting dataset.
The M5 dataset is from the Kaggle M5 Forecasting competition.

Dataset contains:
- sales_train_evaluation.csv: Historical daily sales data
- calendar.csv: Calendar information including events and holidays
- sell_prices.csv: Price information for products

Note:
    The dataset is too large to commit to Git. Users must download it separately.
    
Instructions:
    1. Go to: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
    2. Download the dataset files
    3. Place them in data/raw/ directory
    
Alternative:
    Use Kaggle API to download programmatically (requires API key setup)
"""

import os
import zipfile
from pathlib import Path
from typing import Optional
import subprocess

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def download_m5_with_kaggle_api(output_dir: str = "data/raw") -> bool:
    """
    Download M5 dataset using Kaggle API.
    
    Prerequisites:
        1. Install kaggle: pip install kaggle
        2. Set up Kaggle API credentials:
           - Go to https://www.kaggle.com/account
           - Create API token (downloads kaggle.json)
           - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
           - Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)
    
    Args:
        output_dir: Directory to save downloaded files
        
    Returns:
        True if download successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Attempting to download M5 dataset using Kaggle API...")
    
    try:
        # Check if kaggle is installed
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error("Kaggle CLI not found. Install with: pip install kaggle")
            return False
        
        logger.info("Kaggle CLI found. Downloading dataset...")
        
        # Download the competition data
        download_cmd = [
            "kaggle", "competitions", "download",
            "-c", "m5-forecasting-accuracy",
            "-p", str(output_path)
        ]
        
        result = subprocess.run(
            download_cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            logger.info("You may need to:")
            logger.info("1. Accept competition rules at: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules")
            logger.info("2. Set up Kaggle API credentials")
            return False
        
        logger.info("Download complete. Extracting files...")
        
        # Extract zip file if it exists
        zip_file = output_path / "m5-forecasting-accuracy.zip"
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            logger.info(f"Files extracted to {output_path}")
            
            # Clean up zip file
            zip_file.unlink()
            logger.info("Cleaned up zip file")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False


def check_dataset_exists(data_dir: str = "data/raw") -> dict:
    """
    Check if M5 dataset files exist in the specified directory.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Dictionary with file names as keys and existence status as values
    """
    data_path = Path(data_dir)
    
    required_files = {
        'sales_train_evaluation.csv': False,
        'calendar.csv': False,
        'sell_prices.csv': False
    }
    
    for filename in required_files.keys():
        file_path = data_path / filename
        required_files[filename] = file_path.exists()
    
    return required_files


def print_download_instructions():
    """
    Print instructions for manually downloading the M5 dataset.
    """
    instructions = """
    =====================================================
    M5 DATASET DOWNLOAD INSTRUCTIONS
    =====================================================
    
    The M5 dataset is not included in this repository.
    Please download it using one of these methods:
    
    METHOD 1: Manual Download (Recommended for first-time users)
    -------------------------------------------------------------
    1. Visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
    2. Click "Download All" button (requires Kaggle account)
    3. Extract the downloaded zip file
    4. Copy these files to data/raw/ directory:
       - sales_train_evaluation.csv
       - calendar.csv
       - sell_prices.csv
    
    METHOD 2: Kaggle API (For automated workflows)
    -----------------------------------------------
    1. Install Kaggle CLI: pip install kaggle
    2. Set up API credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token"
       - Place kaggle.json in ~/.kaggle/ directory
    3. Run: python scripts/download_data.py
    
    Required Files:
    ---------------
    - sales_train_evaluation.csv (~140 MB)
    - calendar.csv (~60 KB)
    - sell_prices.csv (~70 MB)
    
    Total Size: ~210 MB
    
    =====================================================
    """
    print(instructions)


if __name__ == "__main__":
    # Check if dataset exists
    file_status = check_dataset_exists()
    
    all_exist = all(file_status.values())
    
    if all_exist:
        logger.info("✓ All M5 dataset files found!")
        for filename, exists in file_status.items():
            logger.info(f"  ✓ {filename}")
    else:
        logger.warning("✗ Some M5 dataset files are missing:")
        for filename, exists in file_status.items():
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {filename}")
        
        print_download_instructions()
        
        # Attempt automatic download
        user_input = input("\nWould you like to try downloading with Kaggle API? (y/n): ")
        if user_input.lower() == 'y':
            success = download_m5_with_kaggle_api()
            if success:
                logger.info("✓ Download complete!")
            else:
                logger.error("✗ Download failed. Please download manually.")
