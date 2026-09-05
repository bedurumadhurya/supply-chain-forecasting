"""
Evaluation Metrics Module

This module implements forecasting evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WMAPE (Weighted Mean Absolute Percentage Error)

WMAPE is particularly important for demand forecasting as it:
- Handles zero values gracefully
- Weights errors by actual demand magnitude
- Provides interpretable percentage error
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
        
    Note:
        MAE is in the same units as the target variable.
        Lower is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
        
    Note:
        RMSE penalizes large errors more than MAE.
        In the same units as target variable.
        Lower is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def wmape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error.
    
    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
    
    This is the industry-standard metric for demand forecasting.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        WMAPE value (as percentage)
        
    Advantages over MAPE:
        - Handles zero values (no division by zero for individual samples)
        - Weights errors by magnitude of actual demand
        - More stable and interpretable
        
    Interpretation:
        - WMAPE of 10% means errors are 10% of total actual demand
        - WMAPE of 20% means errors are 20% of total actual demand
        
    Business Context:
        - < 10%: Excellent
        - 10-20%: Good
        - 20-30%: Acceptable
        - > 30%: Needs improvement
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    
    if denominator == 0:
        logger.warning("Sum of actual values is zero, returning WMAPE=0")
        return 0.0
    
    return (numerator / denominator) * 100


def mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE value (as percentage)
        
    Warning:
        MAPE has issues with zero values and is not recommended for demand forecasting.
        Use WMAPE instead.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    
    if np.sum(mask) == 0:
        logger.warning("All actual values are zero, cannot calculate MAPE")
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'WMAPE': wmape(y_true, y_pred)
    }
    
    return metrics


def wmape_improvement(baseline_wmape: float, model_wmape: float) -> float:
    """
    Calculate percentage improvement over baseline.
    
    Args:
        baseline_wmape: WMAPE of baseline model
        model_wmape: WMAPE of new model
        
    Returns:
        Percentage improvement (positive means better)
        
    Example:
        >>> wmape_improvement(25.0, 20.0)
        20.0  # 20% improvement
    """
    if baseline_wmape == 0:
        return 0.0
    
    improvement = ((baseline_wmape - model_wmape) / baseline_wmape) * 100
    return improvement


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Tracks and compares multiple models on standard metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate a model and store results.
        
        Args:
            model_name: Name of the model
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = calculate_all_metrics(y_true, y_pred)
        
        self.results[model_name] = metrics
        
        logger.info(f"  MAE:   {metrics['MAE']:.4f}")
        logger.info(f"  RMSE:  {metrics['RMSE']:.4f}")
        logger.info(f"  WMAPE: {metrics['WMAPE']:.2f}%")
        
        return metrics
    
    def get_comparison_table(self, baseline_model: str = None) -> pd.DataFrame:
        """
        Get comparison table of all evaluated models.
        
        Args:
            baseline_model: Name of baseline model for improvement calculation
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T
        df = df.round({'MAE': 4, 'RMSE': 4, 'WMAPE': 2})
        
        # Add improvement column if baseline is specified
        if baseline_model and baseline_model in self.results:
            baseline_wmape = self.results[baseline_model]['WMAPE']
            df['Improvement_vs_Baseline'] = df['WMAPE'].apply(
                lambda x: wmape_improvement(baseline_wmape, x)
            ).round(2)
            df['Improvement_vs_Baseline'] = df['Improvement_vs_Baseline'].astype(str) + '%'
        
        # Sort by WMAPE (lower is better)
        df = df.sort_values('WMAPE')
        
        return df
    
    def get_best_model(self) -> tuple:
        """
        Get the best performing model based on WMAPE.
        
        Returns:
            Tuple of (model_name, metrics_dict)
        """
        if not self.results:
            return None, None
        
        best_model = min(self.results.items(), key=lambda x: x[1]['WMAPE'])
        return best_model
    
    def log_summary(self, baseline_model: str = None):
        """
        Log a summary of all model results.
        
        Args:
            baseline_model: Name of baseline model
        """
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        comparison_table = self.get_comparison_table(baseline_model)
        logger.info(f"\n{comparison_table.to_string()}")
        
        best_name, best_metrics = self.get_best_model()
        if best_name:
            logger.info(f"\n✓ Best Model: {best_name}")
            logger.info(f"  WMAPE: {best_metrics['WMAPE']:.2f}%")
            
            if baseline_model and baseline_model in self.results:
                baseline_wmape = self.results[baseline_model]['WMAPE']
                improvement = wmape_improvement(baseline_wmape, best_metrics['WMAPE'])
                logger.info(f"  Improvement over {baseline_model}: {improvement:.2f}%")
        
        logger.info("=" * 60)
