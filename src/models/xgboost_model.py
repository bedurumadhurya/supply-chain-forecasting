"""
XGBoost Forecasting Model

XGBoost is a gradient boosting model that works well for tabular data.
It can capture non-linear relationships and feature interactions.

Key features:
- Handles missing values
- Built-in regularization
- Feature importance
- Fast training
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class XGBoostForecaster:
    """
    XGBoost model for demand forecasting.
    
    This class wraps XGBoost for time series forecasting with:
    - Feature selection
    - Model training
    - Prediction
    - Model persistence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize XGBoost forecaster.
        
        Args:
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config or load_config()
        self.model_config = self.config['models']['xgboost']
        
        self.model = None
        self.feature_columns = None
        self.target_column = 'sales'
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target
            feature_columns: List of feature column names to use
        """
        logger.info("Training XGBoost model...")
        
        # Select features
        if feature_columns is not None:
            self.feature_columns = feature_columns
            X_train_selected = X_train[feature_columns]
            if X_val is not None:
                X_val_selected = X_val[feature_columns]
        else:
            # Use all numeric columns
            self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
            X_train_selected = X_train[self.feature_columns]
            if X_val is not None:
                X_val_selected = X_val[self.feature_columns]
        
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.model_config.get('n_estimators', 100),
            max_depth=self.model_config.get('max_depth', 10),
            learning_rate=self.model_config.get('learning_rate', 0.1),
            subsample=self.model_config.get('subsample', 0.8),
            colsample_bytree=self.model_config.get('colsample_bytree', 0.8),
            random_state=self.model_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Train
        if X_val is not None and y_val is not None:
            logger.info("Training with validation set for early stopping...")
            self.model.fit(
                X_train_selected,
                y_train,
                eval_set=[(X_val_selected, y_val)],
                verbose=False
            )
        else:
            logger.info("Training without validation set...")
            self.model.fit(X_train_selected, y_train)
        
        logger.info("✓ XGBoost training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_selected = X[self.feature_columns]
        predictions = self.model.predict(X_selected)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.model_config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_config = model_data.get('config', {})
        
        logger.info(f"✓ Model loaded from {filepath}")
