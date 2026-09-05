"""
LSTM Forecasting Model

Long Short-Term Memory (LSTM) neural network for time series forecasting.
LSTMs can capture temporal dependencies and long-term patterns.

Key features:
- Sequence-based modeling
- Captures temporal dependencies
- Handles variable-length sequences
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM neural network for forecasting.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            forecast_horizon: Number of steps to forecast
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions (batch_size, forecast_horizon)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(last_hidden)
        
        return output


class LSTMForecaster:
    """
    LSTM forecaster wrapper.
    
    Handles data preparation, training, and prediction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LSTM forecaster.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.model_config = self.config['models']['lstm']
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (n_samples, forecast_horizon)
            X_val: Validation sequences
            y_val: Validation targets
        """
        logger.info("Training LSTM model...")
        
        input_size = X_train.shape[2]
        forecast_horizon = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.model_config.get('hidden_size', 128),
            num_layers=self.model_config.get('num_layers', 2),
            dropout=self.model_config.get('dropout', 0.2),
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config.get('learning_rate', 0.001)
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        epochs = self.model_config.get('epochs', 50)
        batch_size = self.model_config.get('batch_size', 64)
        
        # Simple training loop (production would use DataLoader)
        logger.info(f"Training for {epochs} epochs...")
        
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        logger.info("✓ LSTM training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save(self, filepath: str):
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str, input_size: int, forecast_horizon: int):
        """Load model."""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.model_config.get('hidden_size', 128),
            num_layers=self.model_config.get('num_layers', 2),
            dropout=self.model_config.get('dropout', 0.2),
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(filepath))
        logger.info(f"✓ Model loaded from {filepath}")
