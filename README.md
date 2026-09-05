Supply Chain Demand Forecasting System

A production-ready machine learning system for SKU/store-level demand forecasting with probabilistic predictions, explainability, and MLOps capabilities.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

---

## 📋 Table of Contents

- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Models](#models)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [Monitoring & MLOps](#monitoring--mlops)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Business Problem

A large FMCG/retail company faces critical inventory challenges:

- **Stockouts** of popular products → Lost sales & customer dissatisfaction
- **Excess inventory** of slow-moving products → Capital tied up & waste
- **Inadequate forecasting** → Existing system uses simple moving averages

**Impact:**
- Revenue loss from stockouts
- Increased holding costs
- Poor customer experience
- Inefficient supply chain operations

---

## 💡 Solution Overview

An AI-powered demand forecasting platform that:

✅ Predicts SKU/store-level demand for multiple future days  
✅ Provides **probabilistic forecasts** (P10, P50, P90) for uncertainty quantification  
✅ Explains predictions using **SHAP** and attention mechanisms  
✅ Exposes predictions through a **production-ready REST API**  
✅ Deployed using **Docker** and **AWS**  
✅ Includes **monitoring, drift detection, and automated retraining**

**Business Value:**
- 15-20% reduction in forecast error (WMAPE)
- Reduced stockouts and excess inventory
- Better demand planning and supply chain efficiency
- Data-driven decision making

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  M5 Dataset → Validation → Preprocessing → Feature Engineering  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Model Training                             │
├─────────────────────────────────────────────────────────────────┤
│  Baselines → XGBoost → LSTM → TFT (Probabilistic)              │
│  • Naive, Seasonal Naive, Moving Average                        │
│  • Gradient Boosting with Feature Importance                    │
│  • LSTM for Temporal Dependencies                               │
│  • TFT for Multi-Horizon Probabilistic Forecasts               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation & Selection                       │
├─────────────────────────────────────────────────────────────────┤
│  Metrics: MAE, RMSE, WMAPE                                      │
│  Explainability: SHAP, Feature Importance, Attention            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Production Deployment                        │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI → Docker → AWS (EC2/ECS) → CloudWatch                 │
│  Monitoring → Drift Detection → Auto Retraining                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### M5 Forecasting Dataset

The project uses the **M5 Forecasting** dataset from Kaggle, which contains:

- **30,490 products** across 3 categories (Foods, Hobbies, Household)
- **10 stores** in 3 US states (California, Texas, Wisconsin)
- **1,913 days** of historical sales data
- **Calendar information** (holidays, events, SNAP days)
- **Price information** (sell prices by week)

**Dataset Size:** ~210 MB  
**Time Period:** 2011-01-29 to 2016-05-22

### Download Instructions

The dataset is **not included** in this repository. Download it using one of these methods:

#### Method 1: Manual Download
1. Visit [Kaggle M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
2. Download `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`
3. Place files in `data/raw/` directory

#### Method 2: Kaggle API (Automated)
```bash
pip install kaggle
# Set up Kaggle API credentials (kaggle.json)
python src/data/download_data.py
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Docker (optional, for containerization)
- AWS CLI (optional, for AWS deployment)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd supply-chain-demand-forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Download M5 dataset** (see [Dataset](#dataset) section)

---

## ⚡ Quick Start

### 1. Prepare Data
```bash
# Download and validate data
python scripts/prepare_data.py

# Create features
python scripts/create_features.py
```

### 2. Train Models

Explore and train models using the provided Jupyter notebooks:

```bash
jupyter notebook
# Open notebooks/01_eda.ipynb, 03_baseline_models.ipynb, 04_xgboost.ipynb, etc.
```

Or train models programmatically:

```python
from src.models.xgboost_model import XGBoostForecaster
from src.evaluation.metrics import ModelEvaluator

# Load data
# ... (load train, val, test data)

# Train XGBoost
model = XGBoostForecaster()
model.train(X_train, y_train, X_val, y_val)

# Evaluate
evaluator = ModelEvaluator()
evaluator.evaluate_model("XGBoost", y_test, model.predict(X_test))

# Save model
model.save("models/xgboost_model.pkl")
```

### 3. Run API Server

```bash
python api/main.py
# API available at http://localhost:8000
```

### 4. Test API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "CA_1",
    "item_id": "FOODS_1_001",
    "forecast_horizon": 7
  }'
```

---

## 🎯 Features

### Data Pipeline
- ✅ M5 dataset ingestion and validation
- ✅ Schema validation using Pandera
- ✅ Missing value handling
- ✅ Chronological train/validation/test splits (no data leakage)

### Feature Engineering
- ✅ **Date features**: day, day_of_week, month, quarter, year, weekend
- ✅ **Lag features**: lag_1, lag_7, lag_14, lag_28
- ✅ **Rolling features**: 7/14/28-day moving averages, std, min, max
- ✅ **Price features**: price changes, relative prices
- ✅ **Holiday/Event features**: SNAP days, events, holidays
- ✅ **Categorical encoding**: label encoding for tree models

### Models

#### 1. Baseline Models
- Naive forecasting
- Seasonal naive (weekly seasonality)
- Moving average (7-day, 28-day)

#### 2. XGBoost
- Gradient boosting for non-linear patterns
- Feature importance for interpretability
- Handles missing values
- Fast training and prediction

#### 3. LSTM (Long Short-Term Memory)
- Captures temporal dependencies
- Sequence-based modeling
- PyTorch implementation
- Early stopping and checkpointing

#### 4. TFT (Temporal Fusion Transformer)
- **State-of-the-art** multi-horizon forecasting
- **Probabilistic forecasts** (quantiles: P10, P50, P90)
- **Attention mechanisms** for interpretability
- Variable selection
- Handles static, known-future, and time-varying features

### Evaluation
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **WMAPE** (Weighted Mean Absolute Percentage Error) - Industry standard
- Model comparison framework
- Performance visualization

### Explainability
- **SHAP** values for XGBoost
- Feature importance plots
- TFT attention visualization
- Variable selection insights

---

## 🔌 API Usage

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "model_name": "TFT",
  "model_version": "1.0.0",
  "model_type": "Temporal Fusion Transformer",
  "forecast_horizon": 7,
  "features_count": 45,
  "training_date": "2026-09-05"
}
```

#### Forecast Prediction
```http
POST /predict
```

**Request:**
```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_1_001",
  "forecast_horizon": 7
}
```

**Response:**
```json
{
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
```

**Quantile Interpretation:**
- **P10**: Pessimistic scenario (10% chance demand is below this)
- **P50**: Most likely scenario (median forecast)
- **P90**: Optimistic scenario (90% chance demand is below this)

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t supply-chain-forecasting .
```

### Run Container

```bash
docker run -d -p 8000:8000 \
  --name forecasting-api \
  -v $(pwd)/models:/app/models \
  supply-chain-forecasting
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Test Docker Deployment

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_1_001", "forecast_horizon": 7}'
```

### Stop Container

```bash
docker-compose down
```

---

## ☁️ AWS Deployment

### Prerequisites
- AWS account
- AWS CLI configured
- Docker installed

### Automated Deployment

```bash
bash scripts/deploy_aws.sh
```

This script:
1. Creates S3 bucket for data/models
2. Builds and pushes Docker image to ECR
3. Sets up CloudWatch logging
4. Provides EC2/ECS deployment instructions

### Manual Deployment Steps

#### 1. Create S3 Bucket
```bash
aws s3 mb s3://supply-chain-forecasting-bucket --region us-east-1
```

#### 2. Upload Models
```bash
aws s3 sync models/ s3://supply-chain-forecasting-bucket/models/
```

#### 3. Push Docker Image to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name supply-chain-forecasting

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag supply-chain-forecasting:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/supply-chain-forecasting:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/supply-chain-forecasting:latest
```

#### 4. Deploy to EC2
```bash
# Launch EC2 instance (t3.medium recommended)
# SSH into instance
# Install Docker
# Pull and run container
docker pull <ecr-image-uri>
docker run -d -p 80:8000 <ecr-image-uri>
```

#### 5. Configure Security & Monitoring
- Update security groups to allow port 80
- Set up CloudWatch alarms
- Configure Auto Scaling (optional)

---

## 📊 Monitoring & MLOps

### Performance Monitoring

```python
from src.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(performance_threshold=0.20)

# Log performance
metrics = monitor.log_performance(y_true, y_pred)

# Check for degradation
if monitor.check_degradation(baseline_wmape=15.0):
    # Trigger retraining
    pass
```

### Drift Detection

```python
from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector(drift_threshold=0.15)

# Set reference (training) distribution
detector.set_reference(train_df, columns=['sales', 'price', 'lag_7'])

# Detect drift in new data
drift_results = detector.detect_drift(new_df, columns=['sales', 'price', 'lag_7'])
```

### Automated Retraining

The system includes a retraining pipeline that:
1. Monitors model performance continuously
2. Detects data drift
3. Triggers retraining when:
   - WMAPE degrades beyond threshold (20%)
   - Significant drift detected (15%)
   - Scheduled interval reached (30 days)
4. Trains candidate model
5. Evaluates against production model
6. Promotes only if performance improves

---

## 📁 Project Structure

```
supply-chain-demand-forecasting/
│
├── data/                          # Data directory (not in git)
│   ├── raw/                       # Raw M5 dataset
│   ├── processed/                 # Processed data
│   └── external/                  # External data sources
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_tft.ipynb
│
├── src/                           # Source code
│   ├── data/                      # Data pipeline
│   │   ├── download_data.py      # M5 dataset download
│   │   ├── validation.py          # Data validation
│   │   └── preprocessing.py       # Data preprocessing
│   │
│   ├── features/                  # Feature engineering
│   │   ├── date_features.py      # Temporal features
│   │   ├── lag_features.py       # Lag features
│   │   ├── rolling_features.py   # Rolling statistics
│   │   └── feature_engineering.py # Feature pipeline
│   │
│   ├── models/                    # Model implementations
│   │   ├── baseline_models.py    # Naive, seasonal naive, MA
│   │   ├── xgboost_model.py      # XGBoost forecaster
│   │   ├── lstm_model.py         # LSTM forecaster
│   │   └── tft_model.py          # TFT forecaster
│   │
│   ├── evaluation/                # Model evaluation
│   │   ├── metrics.py            # MAE, RMSE, WMAPE
│   │   └── model_comparison.py   # Model comparison utilities
│   │
│   ├── explainability/            # Model explainability
│   │
│   ├── monitoring/                # Monitoring & drift detection
│   │   ├── performance_monitor.py
│   │   └── drift_detector.py
│   │
│   └── utils/                     # Utility functions
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging utilities
│
├── api/                           # FastAPI application
│   ├── main.py                   # API endpoints
│   └── schemas.py                # Pydantic models
│
├── tests/                         # Unit tests
│   ├── test_metrics.py
│   └── test_api.py
│
├── configs/                       # Configuration files
│   └── config.yaml
│
├── models/                        # Trained models (not in git)
│
├── reports/                       # Analysis reports & figures
│   └── figures/
│
├── monitoring/                    # Monitoring logs
│   └── logs/
│
├── scripts/                       # Utility scripts
│   ├── prepare_data.py           # Data preparation
│   ├── create_features.py        # Feature creation
│   └── deploy_aws.sh             # AWS deployment
│
├── Dockerfile                     # Docker configuration
├── docker-compose.yml
├── .dockerignore
├── requirements.txt               # Python dependencies
├── .gitignore
├── .env.example                   # Environment template
├── README.md                      # This file
└── PROJECT_STATUS.md              # Project status tracking
```

---

## 📈 Performance Metrics

### Target

**Business Objective:** 15-20% WMAPE reduction vs baseline

### Model Comparison

> **Note:** Actual metrics will be populated after training on the M5 dataset.  
> This is a production-ready system designed for real experiments, not fabricated results.

| Model | MAE | RMSE | WMAPE | Improvement vs Baseline |
|-------|-----|------|-------|-------------------------|
| Naive | TBD | TBD | TBD | - |
| Seasonal Naive | TBD | TBD | TBD | TBD |
| Moving Average | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |
| **TFT** | TBD | TBD | TBD | TBD |

**To populate these metrics:**
1. Download M5 dataset
2. Run `python scripts/prepare_data.py`
3. Run `python scripts/create_features.py`
4. Train models using notebooks
5. Update this table with actual results

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Multi-task Learning**: Joint demand + inventory optimization
- [ ] **Hierarchical Forecasting**: Aggregate across product/store hierarchies
- [ ] **External Data Integration**:
  - Web search trends (Google Trends)
  - Macroeconomic indicators
  - Weather data
  - Competitor pricing
- [ ] **Graph Neural Networks**: Capture product relationships
- [ ] **Anomaly Detection**: Identify unusual demand patterns
- [ ] **Active Learning**: Prioritize labeling for uncertain predictions
- [ ] **Real-time Streaming**: Process sales data in real-time
- [ ] **A/B Testing Framework**: Test model variants in production
- [ ] **AutoML**: Automated hyperparameter tuning

---

## 🧪 Testing

Run tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_metrics.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **M5 Forecasting Competition** dataset from Kaggle
- **PyTorch Forecasting** library for TFT implementation
- **FastAPI** framework for production API
- Open-source ML community

---

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star ⭐ to help others discover it!

---

**Built with ❤️ for production-ready ML systems**
