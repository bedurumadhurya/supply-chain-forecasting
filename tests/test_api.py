"""
Tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "forecast_horizon" in data


def test_predict_endpoint():
    """Test prediction endpoint."""
    request_data = {
        "store_id": "CA_1",
        "item_id": "FOODS_1_001",
        "forecast_horizon": 7
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["store_id"] == "CA_1"
    assert data["item_id"] == "FOODS_1_001"
    assert len(data["forecasts"]) == 7
    
    # Check forecast structure
    forecast = data["forecasts"][0]
    assert "date" in forecast
    assert "horizon" in forecast
    assert "p10" in forecast
    assert "p50" in forecast
    assert "p90" in forecast
    
    # Check quantile ordering: p10 <= p50 <= p90
    assert forecast["p10"] <= forecast["p50"]
    assert forecast["p50"] <= forecast["p90"]


def test_predict_endpoint_invalid_horizon():
    """Test prediction endpoint with invalid horizon."""
    request_data = {
        "store_id": "CA_1",
        "item_id": "FOODS_1_001",
        "forecast_horizon": 100  # Too large
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing required fields."""
    request_data = {
        "store_id": "CA_1"
        # Missing item_id
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
