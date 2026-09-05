"""
Configuration Management Module

This module provides utilities for loading and managing configuration settings
from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_env_variables():
    """
    Load environment variables from .env file.
    
    This function loads sensitive configuration like AWS credentials,
    API keys, and database passwords from the .env file.
    
    Note:
        The .env file should never be committed to version control.
    """
    env_file = Path('.env')
    
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try to load from .env.example for development
        example_file = Path('.env.example')
        if example_file.exists():
            print("Warning: .env file not found. Using .env.example for structure reference only.")
            print("Please create a .env file with your actual credentials.")


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract and create Path objects for data directories.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with Path objects for each data directory
    """
    data_config = config.get('data', {})
    
    return {
        'raw': Path(data_config.get('raw_data_path', 'data/raw')),
        'processed': Path(data_config.get('processed_data_path', 'data/processed')),
        'external': Path(data_config.get('external_data_path', 'data/external'))
    }


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Extract configuration for a specific model.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model (e.g., 'xgboost', 'lstm', 'tft')
        
    Returns:
        Dictionary containing model-specific configuration
    """
    models_config = config.get('models', {})
    return models_config.get(model_name, {})


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AWS configuration and merge with environment variables.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing AWS configuration
        
    Note:
        Environment variables take precedence over config file values
        for security reasons.
    """
    aws_config = config.get('aws', {})
    
    # Override with environment variables if present
    aws_config['region'] = os.getenv('AWS_DEFAULT_REGION', aws_config.get('region', 'us-east-1'))
    aws_config['s3_bucket'] = os.getenv('S3_BUCKET_NAME', aws_config.get('s3_bucket', ''))
    
    # Note: AWS credentials should come from environment or IAM roles, never from config files
    aws_config['access_key'] = os.getenv('AWS_ACCESS_KEY_ID')
    aws_config['secret_key'] = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    return aws_config
