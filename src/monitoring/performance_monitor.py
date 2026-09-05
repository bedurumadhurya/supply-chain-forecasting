"""
Performance Monitoring Module

Tracks model performance over time and triggers alerts when performance degrades.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..evaluation.metrics import calculate_all_metrics
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceMonitor:
    """
    Monitor model performance and detect degradation.
    """
    
    def __init__(self, performance_threshold: float = 0.20):
        """
        Initialize performance monitor.
        
        Args:
            performance_threshold: Threshold for performance degradation (e.g., 0.20 = 20%)
        """
        self.performance_threshold = performance_threshold
        self.history = []
    
    def log_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Log model performance metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            timestamp: Timestamp of evaluation
            
        Returns:
            Dictionary with metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = calculate_all_metrics(y_true, y_pred)
        
        record = {
            'timestamp': timestamp,
            **metrics
        }
        
        self.history.append(record)
        
        logger.info(f"Performance logged: WMAPE={metrics['WMAPE']:.2f}%")
        
        return metrics
    
    def check_degradation(self, baseline_wmape: float) -> bool:
        """
        Check if performance has degraded significantly.
        
        Args:
            baseline_wmape: Baseline WMAPE to compare against
            
        Returns:
            True if degradation detected
        """
        if not self.history:
            return False
        
        current_wmape = self.history[-1]['WMAPE']
        degradation = (current_wmape - baseline_wmape) / baseline_wmape
        
        if degradation > self.performance_threshold:
            logger.warning(f"⚠ Performance degradation detected!")
            logger.warning(f"Current WMAPE: {current_wmape:.2f}%")
            logger.warning(f"Baseline WMAPE: {baseline_wmape:.2f}%")
            logger.warning(f"Degradation: {degradation*100:.1f}%")
            return True
        
        return False
    
    def get_history(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)
    
    def save_history(self, filepath: str):
        """Save performance history to CSV."""
        df = self.get_history()
        if not df.empty:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Performance history saved to {filepath}")
