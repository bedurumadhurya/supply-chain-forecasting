"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from src.evaluation.metrics import mae, rmse, wmape, wmape_improvement


def test_mae():
    """Test MAE calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    result = mae(y_true, y_pred)
    expected = 0.5  # mean of [0.5, 0.5, 0, 1]
    
    assert abs(result - expected) < 0.001


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    result = rmse(y_true, y_pred)
    # sqrt(mean([0.25, 0.25, 0, 1])) = sqrt(0.375) ≈ 0.612
    
    assert abs(result - 0.612) < 0.01


def test_wmape():
    """Test WMAPE calculation."""
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 32, 38])
    
    # sum of errors: |2| + |2| + |2| + |2| = 8
    # sum of actuals: 10 + 20 + 30 + 40 = 100
    # WMAPE = 8 / 100 * 100 = 8%
    
    result = wmape(y_true, y_pred)
    assert abs(result - 8.0) < 0.01


def test_wmape_zero_actuals():
    """Test WMAPE with zero actuals."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 2, 3])
    
    result = wmape(y_true, y_pred)
    assert result == 0.0  # Should handle gracefully


def test_wmape_improvement():
    """Test WMAPE improvement calculation."""
    baseline = 25.0
    model = 20.0
    
    improvement = wmape_improvement(baseline, model)
    # (25 - 20) / 25 * 100 = 20%
    
    assert abs(improvement - 20.0) < 0.01


def test_wmape_improvement_worse_model():
    """Test WMAPE improvement with worse model."""
    baseline = 20.0
    model = 25.0
    
    improvement = wmape_improvement(baseline, model)
    # (20 - 25) / 20 * 100 = -25%
    
    assert improvement < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
