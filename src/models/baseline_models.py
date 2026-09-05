"""
Baseline Forecasting Models

Baseline models are essential for benchmarking advanced models.
Every ML forecasting system should compare against these simple methods.

Models implemented:
1. Naive: Last observed value
2. Seasonal Naive: Same day last week
3. Moving Average: Average of last N days
4. Seasonal Moving Average: Average of same weekday
"""

import numpy as np
import pandas as pd
from typing import Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class NaiveForecaster:
    """
    Naive forecasting: Use the last observed value.
    
    This is the simplest baseline. Assumes tomorrow = today.
    """
    
    def __init__(self):
        """Initialize naive forecaster."""
        self.name = "Naive"
    
    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Predict using last observed value.
        
        Args:
            y_history: Historical values
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        last_value = y_history[-1]
        return np.full(horizon, last_value)


class SeasonalNaiveForecaster:
    """
    Seasonal naive forecasting: Use value from same period last season.
    
    For daily data with weekly seasonality: tomorrow = same weekday last week.
    This captures day-of-week patterns.
    """
    
    def __init__(self, season_length: int = 7):
        """
        Initialize seasonal naive forecaster.
        
        Args:
            season_length: Length of season (7 for weekly seasonality)
        """
        self.season_length = season_length
        self.name = f"Seasonal_Naive_{season_length}"
    
    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Predict using seasonal naive method.
        
        Args:
            y_history: Historical values (must be >= season_length)
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if len(y_history) < self.season_length:
            logger.warning(f"History length {len(y_history)} < season length {self.season_length}, using naive forecast")
            return np.full(horizon, y_history[-1])
        
        predictions = []
        for h in range(horizon):
            # Use value from season_length periods ago
            idx = -(self.season_length + h % self.season_length)
            if abs(idx) <= len(y_history):
                predictions.append(y_history[idx])
            else:
                predictions.append(y_history[-1])
        
        return np.array(predictions)


class MovingAverageForecaster:
    """
    Moving average forecasting: Use average of last N observations.
    
    This smooths out noise and provides a stable baseline.
    """
    
    def __init__(self, window: int = 7):
        """
        Initialize moving average forecaster.
        
        Args:
            window: Number of past observations to average
        """
        self.window = window
        self.name = f"Moving_Average_{window}"
    
    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Predict using moving average.
        
        Args:
            y_history: Historical values
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if len(y_history) < self.window:
            avg = np.mean(y_history)
        else:
            avg = np.mean(y_history[-self.window:])
        
        return np.full(horizon, avg)


def generate_baseline_forecasts(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = 'sales',
    forecast_horizon: int = 7
) -> Dict[str, pd.DataFrame]:
    """
    Generate baseline forecasts for all time series in dataset.
    
    Args:
        df: DataFrame with historical data (must be sorted by date)
        group_cols: Columns defining time series groups (e.g., ['store_id', 'item_id'])
        target_col: Target column name
        forecast_horizon: Number of days to forecast
        
    Returns:
        Dictionary of {model_name: forecast_dataframe}
    """
    logger.info("Generating baseline forecasts...")
    
    baselines = {
        'naive': NaiveForecaster(),
        'seasonal_naive': SeasonalNaiveForecaster(season_length=7),
        'moving_avg_7': MovingAverageForecaster(window=7),
        'moving_avg_28': MovingAverageForecaster(window=28)
    }
    
    results = {name: [] for name in baselines.keys()}
    
    # Group by time series
    grouped = df.groupby(group_cols)
    
    for group_name, group_data in grouped:
        y_history = group_data[target_col].values
        
        if len(y_history) < forecast_horizon:
            continue
        
        # Generate forecast with each baseline
        for model_name, model in baselines.items():
            forecast = model.predict(y_history[:-forecast_horizon], horizon=forecast_horizon)
            
            # Store forecast with metadata
            for i, pred in enumerate(forecast):
                results[model_name].append({
                    **dict(zip(group_cols, group_name if isinstance(group_name, tuple) else [group_name])),
                    'horizon': i + 1,
                    'prediction': pred
                })
    
    # Convert to DataFrames
    forecast_dfs = {
        name: pd.DataFrame(forecasts)
        for name, forecasts in results.items()
    }
    
    logger.info(f"✓ Generated forecasts for {len(baselines)} baseline models")
    
    return forecast_dfs
