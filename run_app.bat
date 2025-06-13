@echo off
echo Starting Malaysia Prison Forecasting Application...
echo.
echo Please wait while the application loads...
echo This may take 2-3 minutes on first run while models are trained.
echo.
echo The application will open in your web browser automatically.
echo If it doesn't open, go to: http://localhost:8501
echo.
echo To stop the application, press Ctrl+C in this window.
echo.
streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause