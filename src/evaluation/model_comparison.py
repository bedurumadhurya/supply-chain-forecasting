"""
Model Comparison Module

Utilities for comparing multiple forecasting models and visualizing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import ModelEvaluator
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def plot_predictions(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    dates: Optional[pd.Series] = None,
    title: str = "Model Predictions Comparison",
    save_path: Optional[str] = None
):
    """
    Plot actual vs predicted values for multiple models.
    
    Args:
        y_true: Actual values
        predictions: Dictionary of {model_name: predictions}
        dates: Optional date index for x-axis
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(15, 6))
    
    x = dates if dates is not None else np.arange(len(y_true))
    
    # Plot actual values
    plt.plot(x, y_true, label='Actual', color='black', linewidth=2, alpha=0.7)
    
    # Plot predictions from each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (model_name, y_pred), color in zip(predictions.items(), colors):
        plt.plot(x, y_pred, label=model_name, alpha=0.6, color=color)
    
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Sales')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    evaluator: ModelEvaluator,
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        evaluator: ModelEvaluator with results
        save_path: Path to save figure
    """
    comparison_df = evaluator.get_comparison_table()
    
    if comparison_df.empty:
        logger.warning("No results to plot")
        return
    
    # Remove non-numeric columns
    numeric_cols = ['MAE', 'RMSE', 'WMAPE']
    plot_df = comparison_df[numeric_cols]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(numeric_cols):
        ax = axes[idx]
        plot_df[metric].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.close()


def save_comparison_results(
    evaluator: ModelEvaluator,
    output_path: str,
    baseline_model: str = None
):
    """
    Save model comparison results to CSV.
    
    Args:
        evaluator: ModelEvaluator with results
        output_path: Path to save CSV file
        baseline_model: Name of baseline model
    """
    comparison_df = evaluator.get_comparison_table(baseline_model)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path)
    
    logger.info(f"Comparison results saved to {output_path}")
