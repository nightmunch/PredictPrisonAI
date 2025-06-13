# Malaysia Prison Predictive Planning System

A comprehensive AI-powered forecasting application for prison administrative planning, developed specifically for Malaysian prison system requirements.

## Features

- **Dashboard Overview**: Real-time prison metrics and system status
- **Population Forecasting**: AI-powered predictions with scenario analysis
- **Staffing Planning**: Optimal staff allocation and efficiency planning
- **Resource Management**: Cost forecasting and capacity planning
- **Model Performance**: AI model evaluation and management

## System Requirements

- Windows 10/11
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Installation Instructions

### Option 1: Using Python (Recommended)

1. **Install Python**
   - Download Python 3.11 from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: Open Command Prompt and run `python --version`

2. **Download the Application**
   - Download all files to a folder (e.g., `C:\PrisonForecasting`)
   - Open Command Prompt as Administrator
   - Navigate to the folder: `cd C:\PrisonForecasting`

3. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**
   - Open your web browser
   - Go to `http://localhost:8501`
   - The application will automatically generate data and train models on first run

### Option 2: Using Anaconda (Alternative)

1. **Install Anaconda**
   - Download from [anaconda.com](https://www.anaconda.com/products/distribution)
   - Install with default settings

2. **Create Environment**
   ```bash
   conda create -n prison-forecast python=3.11
   conda activate prison-forecast
   ```

3. **Install Dependencies**
   ```bash
   conda install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
   pip install joblib
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

## Quick Start Guide

1. **First Launch**: The application will automatically generate 7 years of synthetic prison data
2. **Model Training**: AI models will be trained automatically (takes 2-3 minutes)
3. **Navigation**: Use the dropdown menu to switch between different sections
4. **Forecasting**: Adjust parameters in the sidebar to customize predictions
5. **Export**: Download charts and data using the export buttons

## Application Structure

```
prison-forecasting/
├── app.py                    # Main application
├── data/                     # Generated datasets
├── models/                   # Trained AI models
├── modules/                  # Forecasting modules
├── utils/                    # Utility functions
├── .streamlit/               # Configuration
└── README.md                 # This file
```

## Forecasting Capabilities

### Population Forecasting
- 24-month forward predictions
- Scenario analysis (Base Case, Optimistic, Pessimistic, Policy Change)
- Demographic breakdowns by gender, age, and crime type
- Admission and release flow analysis

### Staffing Forecasting
- Staff-to-prisoner ratio optimization
- Category-wise staffing (Security, Admin, Medical, Other)
- Shift distribution analysis
- Overtime and efficiency planning

### Resource Forecasting
- Monthly cost projections
- Capacity utilization analysis
- Energy efficiency optimization
- Infrastructure planning

### Model Performance
- Real-time model accuracy metrics
- Feature importance analysis
- Cross-validation results
- Model retraining capabilities

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Python Not Found**
   - Reinstall Python with "Add to PATH" checked
   - Restart Command Prompt

3. **Module Import Errors**
   ```bash
   pip install --upgrade streamlit pandas numpy scikit-learn plotly
   ```

4. **Slow Model Training**
   - This is normal on first run (2-3 minutes)
   - Subsequent runs will load pre-trained models

### Performance Optimization

- **Memory**: Close other applications if running slowly
- **CPU**: Model training uses all available cores
- **Storage**: Ensure 2GB free space for data and models

## Data Information

The application generates realistic synthetic data including:
- **Population Data**: 84 months of prison population statistics
- **Staffing Data**: Staff allocation across categories and shifts
- **Resource Data**: Operational costs and capacity utilization
- **Demographic Data**: Gender, age, and crime type distributions

All data is automatically generated to reflect Malaysian prison system characteristics.

## Model Information

The application uses ensemble machine learning models:
- **Random Forest**: For non-linear pattern recognition
- **Gradient Boosting**: For complex trend analysis
- **Linear Regression**: For baseline comparisons

Models are automatically evaluated and the best performer is selected for each forecasting type.

## Support and Maintenance

### Regular Maintenance
- Models should be retrained monthly with new data
- System automatically monitors model performance
- Backup data and models folder regularly

### System Updates
- Update Python packages monthly:
  ```bash
  pip install --upgrade streamlit pandas numpy scikit-learn plotly
  ```

### Data Backup
- Important folders to backup:
  - `data/` - Historical datasets
  - `models/` - Trained AI models

## Security Considerations

- Application runs locally on your machine
- No data is sent to external servers
- All processing happens offline
- Suitable for sensitive prison administration data

## Contact Information

For technical support or questions about the forecasting models, refer to the Model Performance section within the application for detailed metrics and validation results.

---

**Note**: This application is designed for administrative planning purposes. All forecasts should be validated against actual operational requirements and policies.