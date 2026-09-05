"""
API Request/Response Schemas

Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import date


class PredictionRequest(BaseModel):
    """
    Request schema for demand forecast prediction.
    """
    store_id: str = Field(..., description="Store identifier (e.g., 'CA_1')")
    item_id: str = Field(..., description="Item/SKU identifier (e.g., 'FOODS_1_001')")
    forecast_horizon: int = Field(7, ge=1, le=28, description="Number of days to forecast (1-28)")
    
    # Optional features for advanced prediction
    features: Optional[Dict[str, float]] = Field(None, description="Additional features")
    
    class Config:
        schema_extra = {
            "example": {
                "store_id": "CA_1",
                "item_id": "FOODS_1_001",
                "forecast_horizon": 7
            }
        }


class ForecastPoint(BaseModel):
    """
    Single forecast point with probabilistic predictions.
    """
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    horizon: int = Field(..., description="Days ahead from prediction date")
    p10: float = Field(..., description="10th percentile prediction")
    p50: float = Field(..., description="50th percentile (median) prediction")
    p90: float = Field(..., description="90th percentile prediction")


class PredictionResponse(BaseModel):
    """
    Response schema for demand forecast prediction.
    """
    store_id: str
    item_id: str
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model")
    forecast_horizon: int
    forecasts: List[ForecastPoint] = Field(..., description="List of forecast points")
    
    class Config:
        schema_extra = {
            "example": {
                "store_id": "CA_1",
                "item_id": "FOODS_1_001",
                "model_name": "TFT",
                "model_version": "1.0.0",
                "forecast_horizon": 7,
                "forecasts": [
                    {
                        "date": "2016-05-23",
                        "horizon": 1,
                        "p10": 2.1,
                        "p50": 3.5,
                        "p90": 5.2
                    },
                    {
                        "date": "2016-05-24",
                        "horizon": 2,
                        "p10": 2.3,
                        "p50": 3.7,
                        "p90": 5.4
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether forecasting model is loaded")


class ModelInfoResponse(BaseModel):
    """
    Model information response.
    """
    model_name: str
    model_version: str
    model_type: str = Field(..., description="Type of model (e.g., 'XGBoost', 'LSTM', 'TFT')")
    forecast_horizon: int
    features_count: int
    training_date: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Model performance metrics (MAE, RMSE, WMAPE)"
    )


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
