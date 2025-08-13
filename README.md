# Malaysia Prison Analytics Powered By Credence AI & Analytics

![Prison Management](https://img.shields.io/badge/Domain-Prison%20Management-blue)
![AI Forecasting](https://img.shields.io/badge/AI-Forecasting-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Bilingual](https://img.shields.io/badge/Language-EN%20%7C%20BM-brightgreen)

A professional AI-powered prison analytics dashboard designed for the Malaysian Prison Department. This system provides comprehensive forecasting and analysis for prison population, staffing requirements, and resource planning with full bilingual support (English/Bahasa Malaysia) and official government branding.

## 🎯 Project Overview

The Malaysia Prison Analytics system provides professional-grade analytics for prison administration with:

- **Population Forecasting**: AI-powered prediction of prison population trends and patterns
- **Staffing Analytics**: Optimal staff allocation and requirement planning
- **Resource Management**: Comprehensive cost analysis and budget forecasting
- **Bilingual Interface**: Complete English and Bahasa Malaysia localization
- **Official Branding**: Professional Malaysian Prison Department styling

### ✨ Key Features

- 🤖 **Advanced AI Models**: Machine learning forecasting with high accuracy
- 🌐 **Bilingual Support**: Complete English/Bahasa Malaysia interface
- 🏛️ **Government Standards**: Official Malaysian Prison Department branding
- 📊 **Interactive Dashboard**: Professional Streamlit-based web interface
- 📈 **Real-time Analytics**: Live data processing and visualization
- � **Realistic Data**: Accurate Malaysian prison statistics (RM 15/day per prisoner)
- 🔒 **Secure Deployment**: Complete offline operation for data security
- 📋 **Comprehensive Reports**: Detailed insights for decision makers

## 🚀 Quick Start

### Prerequisites

- Windows 10/11 or macOS/Linux
- Python 3.8+ (Python 3.11 recommended)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/RajKhalifaa/PredictPrisonAI.git
cd PredictPrisonAI
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run Application

```bash
streamlit run app.py
```

#### 4. Access Dashboard

- Open your browser to: **http://localhost:8501**
- Use the language selector in the sidebar to switch between English and Bahasa Malaysia
- Navigate through different analytics modules using the sidebar menu

### 🌐 Language Support

The application supports complete bilingual interface:

- **English**: Default interface language
- **Bahasa Malaysia**: Complete localization including charts, navigation, and data labels
- Switch languages anytime using the sidebar selector

## 📁 Project Structure

```
PrisonPredictAI/
├── 📄 app.py                          # Main Streamlit application
├── 📄 data_generator.ipynb            # Data generation notebook
├── 📄 pyproject.toml                  # Project configuration
├── 📄 uv.lock                         # Dependency lock file
├── 📄 .gitignore                      # Git ignore rules
├──
├── 📁 assets/                         # Static assets
│   └── 📄 Penjara-logo.jpg           # Official Malaysian Prison logo
├──
├── 📁 data/                           # Data files (60 monthly records)
│   ├── 📄 malaysia_prisons.json      # Prison locations and details
│   ├── 📄 population_data.csv        # Historical population data (2020-2024)
│   ├── 📄 prison_detail_data.csv     # Detailed prison information
│   ├── 📄 resource_data.csv          # Resource allocation data
│   └── 📄 staffing_data.csv          # Staffing information
├──
├── 📁 models/                         # Machine learning models
│   ├── 📄 model_trainer.py           # Model training utilities
│   ├── 📄 feature_importance.pkl     # Feature importance data
│   ├── 📄 model_metrics.pkl          # Model performance metrics
│   ├── 📄 population_model.pkl       # Population forecasting model
│   ├── 📄 resource_model.pkl         # Resource forecasting model
│   └── 📄 staffing_model.pkl         # Staffing forecasting model
├──
├── 📁 modules/                        # Application modules
│   ├── 📄 __init__.py                # Package initialization
│   ├── 📄 model_performance.py       # Model performance analysis
│   ├── 📄 population_forecast.py     # Population forecasting module
│   ├── 📄 resource_forecast.py       # Resource forecasting module
│   └── 📄 staffing_forecast.py       # Staffing forecasting module
├──
└── 📁 utils/                          # Utility functions
    ├── 📄 __init__.py                # Package initialization
    ├── 📄 data_utils.py              # Data processing utilities
    └── 📄 visualization.py           # Chart and visualization utilities
```

## 🔧 System Features

### 📊 Population Analytics

- **5-Year Historical Data**: Monthly records from 2020-2024 (60 data points)
- **Population Breakdown**: Gender, age groups, crime types
- **Trend Analysis**: Admission and release patterns
- **Capacity Planning**: Current vs. maximum capacity analysis

### 👥 Staffing Management

- **Staff Optimization**: Intelligent staff-to-prisoner ratio calculations
- **Category Analysis**: Security, administrative, medical, and support staff
- **Cost Planning**: Salary projections and budget forecasting
- **Efficiency Metrics**: Performance tracking and optimization

### 💰 Resource Planning

- **Realistic Costs**: RM 15/day per prisoner (Malaysian standard)
- **Monthly Budgets**: RM 21-29 million operational costs
- **Resource Allocation**: Food, medical, utilities, maintenance
- **Inflation Modeling**: Adjustable economic factors

### 🎨 Professional Interface

- **Official Branding**: Malaysian Prison Department logo and styling
- **Government Standards**: Professional color schemes and typography
- **Responsive Design**: Works on desktop and mobile devices
- **Accessibility**: User-friendly navigation and clear information hierarchy

## 📈 Data Overview

The system includes comprehensive Malaysian prison data:

- **Total Population**: ~52,000 prisoners (realistic for Malaysia)
- **Prison Capacity**: ~65,000 maximum capacity
- **Monthly Costs**: RM 21-29 million (RM 15/day per prisoner)
- **Time Series**: 60 monthly records (Jan 2020 - Dec 2024)
- **Forecasting**: 6-24 month predictions with scenario analysis

## 🌍 Localization

Complete bilingual support includes:

- **Navigation menus** in English and Bahasa Malaysia
- **Chart labels and titles** fully translated
- **Data descriptions** and tooltips localized
- **Error messages** and notifications in both languages
- **Professional terminology** accurate for Malaysian government use
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

## 🤖 Machine Learning Models

The system uses advanced AI models for accurate forecasting:

### Model Performance

- **Population Model**: 82-83% accuracy, ±713 prisoners RMSE
- **Staffing Model**: 87-90% accuracy, ±329 staff RMSE
- **Resource Model**: 98-99% accuracy, ±464K MYR RMSE

### Algorithms Used

- **Random Forest**: Primary forecasting algorithm
- **Gradient Boosting**: Ensemble learning for accuracy
- **Linear Regression**: Baseline model for comparison
- **Cross-Validation**: 5-fold validation for model stability

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Professional web interface)
- **Backend**: Python 3.8+ with scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Random Forest, Gradient Boosting
- **Deployment**: Local/Server deployment ready

## 📋 Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
plotly>=5.11.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```

## � Deployment

### Local Development

```bash
streamlit run app.py
```

### Server Deployment

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 📞 Support

For technical support or questions about the Malaysia Prison Analytics system:

- **Repository**: [https://github.com/RajKhalifaa/PredictPrisonAI](https://github.com/RajKhalifaa/PredictPrisonAI)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Documentation**: Comprehensive inline documentation available in the code

## 📄 License

This project is developed for the Malaysian Prison Department and contains realistic Malaysian prison data and official government branding.

---

**Malaysia Prison Analytics Powered By Credence AI & Analytics**  
_Professional prison analytics for data-driven decision making_

**3. Model Training Errors**

```bash
# Clear models and regenerate
rm -rf models/*.pkl data/*.csv

# Restart application (will auto-regenerate)
streamlit run app.py
```
