# Malaysia Prison Predictive Planning System

![Prison Management](https://img.shields.io/badge/Domain-Prison%20Management-blue)
![AI Forecasting](https://img.shields.io/badge/AI-Forecasting-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)

A comprehensive AI-powered predictive planning application designed specifically for Malaysia prison administration. This system uses advanced machine learning algorithms to forecast prison population, staffing requirements, and resource needs, enabling data-driven decision making for strategic planning and operational efficiency.

## 🎯 Project Overview

The Malaysia Prison Predictive Planning System addresses critical challenges in prison administration by providing accurate forecasts and scenario analysis for:

- **Population Management**: Predict future prison population with different crime rate scenarios
- **Staffing Optimization**: Calculate optimal staff requirements based on population forecasts
- **Resource Planning**: Forecast operational costs, infrastructure needs, and budget requirements
- **Strategic Planning**: Support long-term capacity planning and policy decision making

### Key Features

- 🤖 **AI-Powered Forecasting**: Advanced machine learning models (Random Forest, Gradient Boosting, Linear Regression)
- 📊 **Interactive Dashboard**: User-friendly Streamlit interface with real-time visualizations
- 🌐 **Multi-Language Support**: English and Malay (Bahasa Malaysia) via sidebar selector in a single app
- 🔍 **Scenario Analysis**: Compare optimistic, pessimistic, base case, and policy change scenarios
- 📈 **Performance Monitoring**: Model accuracy tracking and performance metrics
- 🔒 **Offline Operation**: Complete local deployment for data security and privacy
- 📋 **Comprehensive Reporting**: Detailed insights and recommendations for administrators

## 🚀 Quick Start (Windows + VS Code)

### Prerequisites

- Windows 10/11
- Python 3.8+ (Python 3.11 recommended)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Installation Steps

#### 1. Install Python

```bash
# Download Python 3.11 from https://www.python.org/downloads/
# During installation, CHECK "Add Python to PATH"
# Verify installation:
python --version
```

#### 2. Clone Repository

```bash
git clone https://github.com/yourusername/malaysia-prison-forecasting.git
cd malaysia-prison-forecasting
```

#### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Run Application

```bash
streamlit run app.py
```

#### 6. Access Application

- Open the displayed URL: **http://localhost:8501**
- On first run, data is generated and models are trained (2-3 minutes)
- Use the sidebar to navigate between dashboard pages
- **Language Options:**
  - Use the **Language** selector in the sidebar to switch between English and Malay (Bahasa Malaysia)

#### 7. For VM/Network Hosting (Optional)

```bash
streamlit run app.py --server.address 0.0.0.0
```

- Access from other machines: **http://[YOUR-IP]:8501**
- Find your IP with: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)

## 📁 Project Structure

```
PrisonPredictAI/
├── app.py                      # Main Streamlit application (English & Malay)
├── modules/                    # Forecasting modules
│   ├── __init__.py
│   ├── population_forecast.py  # Population prediction module
│   ├── staffing_forecast.py    # Staffing optimization module
│   ├── resource_forecast.py    # Resource planning module
│   └── model_performance.py    # Model evaluation module
├── models/                     # AI models and training
│   ├── model_trainer.py        # ML model training logic
│   ├── population_model.pkl    # Trained population model
│   ├── staffing_model.pkl      # Trained staffing model
│   ├── resource_model.pkl      # Trained resource model
│   ├── model_metrics.pkl       # Performance metrics
│   └── feature_importance.pkl  # Feature importance data
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_utils.py           # Data generation and management
│   └── visualization.py        # Chart and plot functions
├── data/                       # Generated datasets (auto-created)
│   ├── population_data.csv     # Prison population data
│   ├── staffing_data.csv       # Staffing information
│   └── resource_data.csv       # Resource and cost data
├── .streamlit/                 # Streamlit configuration
│   └── config.toml             # Server and UI settings
├── requirements.txt            # Python dependencies list
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔧 System Capabilities

### 1. Population Forecasting

- **Forecast Horizon**: 6-36 months
- **Scenario Types**:
  - **Base Case**: Normal trend continuation
  - **Optimistic**: Lower crime rates, better rehabilitation
  - **Pessimistic**: Higher crime rates, policy changes
  - **Policy Change**: Specific legislative impacts
- **Demographics**: Gender, age groups, crime type analysis
- **Flow Analysis**: Admission and release patterns

### 2. Staffing Forecasting

- **Staff-to-Prisoner Ratios**: Configurable target ratios (0.20-0.40)
- **Staff Categories**: Security, Administrative, Medical, Other
- **Efficiency Modeling**: Technology and process improvements
- **Operational Metrics**: Overtime, availability, shift analysis
- **Cost Calculation**: Salary projections and recruitment planning

### 3. Resource Forecasting

- **Cost Components**: Food, medical, utilities, maintenance
- **Inflation Modeling**: Adjustable annual inflation rates
- **Efficiency Targets**: Energy savings and waste reduction
- **Capacity Planning**: Infrastructure expansion needs
- **Budget Optimization**: Cost reduction recommendations

### 4. Model Performance

- **Algorithm Types**: Random Forest, Gradient Boosting, Linear Regression
- **Accuracy Metrics**: R², RMSE, MAE, Cross-validation scores
- **Feature Importance**: Key predictive factors identification
- **Model Selection**: Automatic best performer selection

## 📊 Data Generation

The system automatically generates 7 years (84 months) of realistic synthetic data:

### Population Data

- **Base Population**: 75,000 prisoners
- **Growth Pattern**: 5,000 increase over 7 years with seasonal variation
- **Demographics**: 85% male, age distribution (25% young, 55% middle, 20% older)
- **Crime Types**: Drug crimes (35%), violent (25%), property (20%), other (20%)
- **Flow Metrics**: ~3,000 monthly admissions, ~2,800 monthly releases

### Staffing Data

- **Total Staff**: ~21,000 employees (0.28 staff-to-prisoner ratio)
- **Distribution**: Security (65%), Admin (15%), Medical (8%), Other (12%)
- **Operational**: 120 hours/month overtime, 8% sick leave, 12% vacation

### Resource Data

- **Daily Cost**: MYR 45 per prisoner
- **Cost Breakdown**: Food (40%), Medical (15%), Utilities (20%), Other (25%)
- **Capacity**: 95,000 maximum, 75-85% utilization
- **Efficiency**: 75% energy efficiency, 12% food waste rate

## 🤖 Machine Learning Models

### Model Training Process

1. **Feature Engineering**: Lag features, rolling statistics, seasonal patterns
2. **Algorithm Competition**: Random Forest vs Gradient Boosting vs Linear Regression
3. **Performance Evaluation**: RMSE-based model selection
4. **Cross-Validation**: 5-fold validation for stability testing
5. **Feature Importance**: Tree-based importance ranking

### Current Performance Metrics

- **Population Model**: 82-83% R², ±713 prisoners RMSE
- **Staffing Model**: 87-90% R², ±329 staff RMSE
- **Resource Model**: 98-99% R², ±464K MYR RMSE

### Model Features

- **Time Series**: Previous values, trends, seasonality
- **Cross-Dependencies**: Population → Staffing → Resources
- **External Factors**: Policy changes, efficiency improvements

## ⚙️ Configuration Options

### Streamlit Configuration (.streamlit/config.toml)

```toml
[server]
headless = false
address = "localhost"
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"

[browser]
gatherUsageStats = false
```

## 🛠️ Development Workflow

### Daily Development

```bash
# Activate environment
venv\Scripts\activate

# Run the application (local access)
streamlit run app.py

# Run with network access for VM hosting
streamlit run app.py --server.address 0.0.0.0
```

### Model Retraining

```python
# Automatic retraining (happens on data changes)
# Manual retraining
from models.model_trainer import train_all_models
from utils.data_utils import load_or_generate_data

data = load_or_generate_data()
models = train_all_models(data)
```

## 🚨 Troubleshooting

### Common Issues

**1. Python/Module Not Found**

```bash
# Ensure Python is in PATH
python --version

# Reinstall dependencies
pip install --upgrade streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

**2. Port Already in Use**

```bash
# Use different port
streamlit run app.py --server.port 8502

# Kill existing processes
taskkill /f /im python.exe
```

**3. Model Training Errors**

```bash
# Clear models and regenerate
rm -rf models/*.pkl data/*.csv

# Restart application (will auto-regenerate)
streamlit run app.py
```
