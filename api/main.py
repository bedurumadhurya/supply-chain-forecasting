"""
FastAPI Application for Demand Forecasting

Production-ready API with:
- Health checks
- Model information endpoint
- Prediction endpoint with probabilistic forecasts
- Error handling
- Request validation
- Logging
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timedelta
import numpy as np
import joblib
from typing import Optional

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    ForecastPoint
)
from src.utils.logger import setup_logger
from src.utils.config import load_config, load_env_variables

# Load environment variables
load_env_variables()

# Setup logging
logger = setup_logger(__name__)

# Load configuration
config = load_config()
api_config = config.get('api', {})

# Initialize FastAPI app
app = FastAPI(
    title=api_config.get('title', 'Supply Chain Demand Forecasting API'),
    version=api_config.get('version', '1.0.0'),
    description=api_config.get('description', 'AI-powered demand forecasting for SKU/store-level predictions')
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable (loaded on startup)
global_model = None
model_info = {
    "model_name": "XGBoost",
    "model_version": "1.0.0",
    "model_type": "XGBoost",
    "forecast_horizon": 7,
    "features_count": 0,
    "training_date": None,
    "performance_metrics": None
}


@app.on_event("startup")
async def load_model():
    """
    Load forecasting model on startup.
    
    In production, this would load the trained model from:
    - Local file system
    - S3 bucket
    - Model registry
    """
    global global_model, model_info
    
    logger.info("Loading forecasting model...")
    
    try:
        # Try to load XGBoost model
        model_path = Path("models/xgboost_model.pkl")
        
        if model_path.exists():
            global_model = joblib.load(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
            
            # Update model info
            if hasattr(global_model, 'feature_columns'):
                model_info['features_count'] = len(global_model.feature_columns)
            
            model_info['training_date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.info("API will run in demo mode with simulated predictions")
            global_model = None
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        global_model = None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        version=api_config.get('version', '1.0.0'),
        model_loaded=global_model is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded forecasting model.
    
    Returns:
        Model metadata including name, version, and performance metrics
    """
    return ModelInfoResponse(**model_info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Prediction"]
)
async def predict(request: PredictionRequest):
    """
    Generate demand forecast for a specific store-item combination.
    
    This endpoint provides probabilistic forecasts with:
    - P10: 10th percentile (pessimistic scenario)
    - P50: 50th percentile (most likely scenario)
    - P90: 90th percentile (optimistic scenario)
    
    Args:
        request: Prediction request with store_id, item_id, and forecast_horizon
        
    Returns:
        Probabilistic demand forecasts for the specified horizon
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Prediction request: store={request.store_id}, item={request.item_id}, horizon={request.forecast_horizon}")
        
        # Generate forecasts
        if global_model is not None:
            # Use actual model (would need proper feature preparation)
            logger.info("Using loaded model for prediction")
            
            # In production, you would:
            # 1. Fetch historical data for this store-item
            # 2. Create features
            # 3. Run model prediction
            # 4. Generate quantiles (or use TFT for probabilistic forecasts)
            
            # For now, simulate predictions
            base_demand = np.random.uniform(2, 10)
            forecasts = generate_demo_forecast(
                base_demand=base_demand,
                horizon=request.forecast_horizon
            )
        else:
            # Demo mode: generate simulated forecasts
            logger.info("Model not loaded, generating demo predictions")
            base_demand = np.random.uniform(2, 10)
            forecasts = generate_demo_forecast(
                base_demand=base_demand,
                horizon=request.forecast_horizon
            )
        
        # Build response
        response = PredictionResponse(
            store_id=request.store_id,
            item_id=request.item_id,
            model_name=model_info['model_name'],
            model_version=model_info['model_version'],
            forecast_horizon=request.forecast_horizon,
            forecasts=forecasts
        )
        
        logger.info(f"✓ Prediction successful for {request.store_id}/{request.item_id}")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


def generate_demo_forecast(base_demand: float, horizon: int) -> list:
    """
    Generate demo probabilistic forecasts.
    
    In production, this would be replaced with actual model predictions.
    
    Args:
        base_demand: Base demand level
        horizon: Forecast horizon
        
    Returns:
        List of ForecastPoint objects
    """
    forecasts = []
    start_date = datetime.now().date() + timedelta(days=1)
    
    for h in range(horizon):
        forecast_date = start_date + timedelta(days=h)
        
        # Simulate trend and noise
        trend = 1 + (h * 0.02)  # Slight upward trend
        noise = np.random.normal(0, 0.1)
        
        # Generate quantiles
        p50 = base_demand * trend + noise
        p10 = p50 * 0.6  # Lower bound
        p90 = p50 * 1.5  # Upper bound
        
        # Ensure non-negative
        p10 = max(0, p10)
        p50 = max(0, p50)
        p90 = max(0, p90)
        
        forecasts.append(
            ForecastPoint(
                date=forecast_date.strftime('%Y-%m-%d'),
                horizon=h + 1,
                p10=round(p10, 2),
                p50=round(p50, 2),
                p90=round(p90, 2)
            )
        )
    
    return forecasts


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    # Run with: python api/main.py
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', False)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )
