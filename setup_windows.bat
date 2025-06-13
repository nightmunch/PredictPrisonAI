@echo off
echo ========================================
echo Malaysia Prison Forecasting Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Installing required packages...
echo.

pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib

if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    echo Please run this script as Administrator
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To run the application:
echo 1. Open Command Prompt
echo 2. Navigate to this folder
echo 3. Run: streamlit run app.py
echo.
echo The application will open in your web browser at:
echo http://localhost:8501
echo.
pause