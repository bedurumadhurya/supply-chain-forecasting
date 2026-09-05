"""
Temporal Fusion Transformer (TFT) Model

TFT is a state-of-the-art model for multi-horizon forecasting with:
- Attention mechanisms
- Variable selection
- Probabilistic forecasting (quantile predictions)
- Interpretability

This implementation uses PyTorch Forecasting library.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    import pytorch_lightning as pl
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class TFTForecaster:
    """
    Temporal Fusion Transformer forecaster.
    
    Provides multi-horizon probabilistic forecasting with interpretability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TFT forecaster.
        
        Args:
            config: Configuration dictionary
        """
        if not TFT_AVAILABLE:
            logger.warning("PyTorch Forecasting not available. TFT model cannot be used.")
            logger.warning("Install with: pip install pytorch-forecasting")
        
        self.config = config or load_config()
        self.model_config = self.config['models']['tft']
        
        self.model = None
        self.training_dataloader = None
        self.validation_dataloader = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        time_idx: str = 'time_idx',
        target: str = 'sales',
        group_ids: List[str] = None,
        static_categoricals: List[str] = None,
        time_varying_known_categoricals: List[str] = None,
        time_varying_known_reals: List[str] = None,
        time_varying_unknown_reals: List[str] = None,
        max_encoder_length: int = 28,
        max_prediction_length: int = 7
    ) -> TimeSeriesDataSet:
        """
        Prepare data for TFT training.
        
        Args:
            df: DataFrame with time series data
            time_idx: Column name for time index
            target: Target column name
            group_ids: Columns defining time series groups
            static_categoricals: Static categorical features
            time_varying_known_categoricals: Known future categorical features
            time_varying_known_reals: Known future real-valued features
            time_varying_unknown_reals: Unknown future real-valued features
            max_encoder_length: Length of encoder (lookback)
            max_prediction_length: Length of prediction horizon
            
        Returns:
            TimeSeriesDataSet for training
        """
        if not TFT_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available")
        
        # Default group IDs
        if group_ids is None:
            group_ids = ['store_id', 'item_id']
        
        # Create time series dataset
        training = TimeSeriesDataSet(
            df,
            time_idx=time_idx,
            target=target,
            group_ids=group_ids,
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals or [],
            time_varying_known_categoricals=time_varying_known_categoricals or [],
            time_varying_known_reals=time_varying_known_reals or [],
            time_varying_unknown_reals=time_varying_unknown_reals or [],
            target_normalizer=GroupNormalizer(
                groups=group_ids, transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        
        return training
    
    def train(
        self,
        training_data: TimeSeriesDataSet,
        validation_data: Optional[TimeSeriesDataSet] = None,
        max_epochs: int = 50
    ):
        """
        Train TFT model.
        
        Args:
            training_data: Training TimeSeriesDataSet
            validation_data: Validation TimeSeriesDataSet
            max_epochs: Maximum training epochs
        """
        if not TFT_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available")
        
        logger.info("Training TFT model...")
        
        # Create dataloaders
        batch_size = self.model_config.get('batch_size', 128)
        self.training_dataloader = training_data.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        
        if validation_data is not None:
            self.validation_dataloader = validation_data.to_dataloader(
                train=False, batch_size=batch_size * 10, num_workers=0
            )
        
        # Initialize TFT model
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=self.model_config.get('learning_rate', 0.001),
            hidden_size=self.model_config.get('hidden_size', 64),
            attention_head_size=self.model_config.get('attention_head_size', 4),
            dropout=self.model_config.get('dropout', 0.1),
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=self.model_config.get('quantiles', [0.1, 0.5, 0.9])),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            gradient_clip_val=0.1,
            limit_train_batches=30,  # Limit for faster training
            callbacks=[],
            logger=False
        )
        
        # Train
        trainer.fit(
            self.model,
            train_dataloaders=self.training_dataloader,
            val_dataloaders=self.validation_dataloader
        )
        
        logger.info("✓ TFT training complete")
    
    def predict(
        self,
        data: TimeSeriesDataSet,
        return_quantiles: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make probabilistic predictions.
        
        Args:
            data: TimeSeriesDataSet for prediction
            return_quantiles: Whether to return quantile predictions
            
        Returns:
            Dictionary with predictions and quantiles
        """
        if not TFT_AVAILABLE or self.model is None:
            raise ValueError("Model not trained")
        
        dataloader = data.to_dataloader(train=False, batch_size=128, num_workers=0)
        
        # Get predictions
        predictions = self.model.predict(dataloader, return_x=False)
        
        if return_quantiles and len(predictions.shape) > 2:
            # predictions shape: (n_samples, horizon, n_quantiles)
            return {
                'p10': predictions[:, :, 0],
                'p50': predictions[:, :, 1],
                'p90': predictions[:, :, 2]
            }
        else:
            return {'predictions': predictions}
    
    def save(self, filepath: str):
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model."""
        if not TFT_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available")
        
        forecaster = cls()
        forecaster.model = TemporalFusionTransformer.load_from_checkpoint(filepath)
        logger.info(f"✓ Model loaded from {filepath}")
        return forecaster
