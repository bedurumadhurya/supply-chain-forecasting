"""
Data Drift Detection Module

Detects distribution shifts in input features and target variable.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DriftDetector:
    """
    Detect data drift using statistical tests.
    """
    
    def __init__(self, drift_threshold: float = 0.15):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for drift detection
        """
        self.drift_threshold = drift_threshold
        self.reference_distributions = {}
    
    def set_reference(self, df: pd.DataFrame, columns: List[str]):
        """
        Set reference distributions for drift detection.
        
        Args:
            df: Reference DataFrame (training data)
            columns: Columns to monitor
        """
        for col in columns:
            if col in df.columns:
                self.reference_distributions[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        logger.info(f"Reference distributions set for {len(self.reference_distributions)} columns")
    
    def detect_drift(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """
        Detect drift in specified columns.
        
        Args:
            df: Current DataFrame
            columns: Columns to check for drift
            
        Returns:
            Dictionary with drift results
        """
        drift_results = {}
        
        for col in columns:
            if col not in self.reference_distributions or col not in df.columns:
                continue
            
            ref = self.reference_distributions[col]
            
            # Calculate current statistics
            current_mean = df[col].mean()
            current_std = df[col].std()
            
            # Calculate drift (relative change in mean)
            if ref['mean'] != 0:
                mean_drift = abs(current_mean - ref['mean']) / abs(ref['mean'])
            else:
                mean_drift = 0
            
            # Check if drift exceeds threshold
            has_drift = mean_drift > self.drift_threshold
            
            drift_results[col] = {
                'drift_score': mean_drift,
                'has_drift': has_drift,
                'reference_mean': ref['mean'],
                'current_mean': current_mean
            }
            
            if has_drift:
                logger.warning(f"⚠ Drift detected in {col}:")
                logger.warning(f"  Reference mean: {ref['mean']:.4f}")
                logger.warning(f"  Current mean: {current_mean:.4f}")
                logger.warning(f"  Drift score: {mean_drift:.2%}")
        
        return drift_results
    
    def summary(self, drift_results: Dict) -> Dict:
        """
        Summarize drift detection results.
        
        Args:
            drift_results: Results from detect_drift()
            
        Returns:
            Summary dictionary
        """
        total_features = len(drift_results)
        drifted_features = sum(1 for r in drift_results.values() if r['has_drift'])
        
        return {
            'total_features_monitored': total_features,
            'features_with_drift': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0
        }
