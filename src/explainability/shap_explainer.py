"""
SHAP Explainability Module

Provides model interpretability using SHAP (SHapley Additive exPlanations).
SHAP values show the contribution of each feature to individual predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainer for XGBoost models.
    
    SHAP provides:
    - Feature importance
    - Individual prediction explanations
    - Feature interactions
    """
    
    def __init__(self, model, feature_names: list):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (XGBoost)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def fit(self, X_background: np.ndarray, sample_size: int = 100):
        """
        Fit SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP (typically training data sample)
            sample_size: Number of background samples to use
        """
        logger.info("Initializing SHAP explainer...")
        
        # Sample background data for efficiency
        if len(X_background) > sample_size:
            indices = np.random.choice(len(X_background), sample_size, replace=False)
            X_background = X_background[indices]
        
        # Create TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model, X_background)
        
        logger.info("✓ SHAP explainer initialized")
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for predictions.
        
        Args:
            X: Input features
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def plot_summary(self, save_path: Optional[str] = None):
        """
        Plot SHAP summary showing feature importance.
        
        Args:
            save_path: Path to save the figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain() first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance based on mean absolute SHAP values.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain() first.")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return importance_df
    
    def explain_prediction(
        self,
        instance_idx: int,
        X: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Explain a single prediction with waterfall plot.
        
        Args:
            instance_idx: Index of the instance to explain
            X: Input features
            save_path: Path to save the figure
        """
        if self.shap_values is None:
            self.explain(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=X[instance_idx],
            feature_names=self.feature_names
        )
        
        # Waterfall plot
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction explanation saved to {save_path}")
        
        plt.show()
