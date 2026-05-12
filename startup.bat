@echo off
REM ============================================================================
REM Yelp ML Project - Startup Script for Windows
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════════╗
echo ║                  🍽️  YELP ML PIPELINE - STARTUP MENU                  ║
echo ╚══════════════════════════════════════════════════════════════════════════╝
echo.

echo Choose an option:
echo.
echo 1) RUN FULL PIPELINE (Train Model + Visualizations + Report)
echo 2) Train ML Model Only
echo 3) Generate Visualizations Only
echo 4) Launch Professional Dashboard
echo 5) View Model Metrics
echo 6) Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo ⏳ Running full pipeline...
    echo.
    python run_full_pipeline.py
    echo.
    echo ✅ Pipeline complete! View outputs/ folder for results.
    echo.
    pause
) else if "%choice%"=="2" (
    echo.
    echo ⏳ Training model (this may take a while)...
    echo.
    python advanced_yelp_ml.py
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    echo ⏳ Generating visualizations...
    echo.
    python ml_visualizations.py
    echo.
    echo ✅ Visualizations saved to outputs/visualizations/
    echo.
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🚀 Starting Streamlit dashboard...
    echo.
    echo Dashboard will open in your browser at http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run dashboard_professional.py
) else if "%choice%"=="5" (
    echo.
    echo 📊 Model Metrics:
    echo.
    if exist outputs\metrics.txt (
        type outputs\metrics.txt
    ) else (
        echo ❌ Metrics file not found. Run the pipeline first.
    )
    echo.
    pause
) else if "%choice%"=="6" (
    echo.
    echo Goodbye! 👋
    echo.
    exit /b 0
) else (
    echo.
    echo ❌ Invalid choice. Please try again.
    echo.
    pause
    cls
    goto :eof
)
