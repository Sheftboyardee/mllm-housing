@echo off
REM Script to start both FastAPI and Streamlit together on Windows

echo ========================================
echo Starting MLLM House Search Application
echo ========================================
echo.

echo [1/2] Starting FastAPI server on port 8000...
start "FastAPI Server" cmd /k "uvicorn app.api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for FastAPI to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit app on port 8501...
start "Streamlit App" cmd /k "streamlit run app/streamlit_app.py"

echo.
echo ========================================
echo Services Started!
echo ========================================
echo.
echo FastAPI Server:  http://localhost:8000
echo API Docs:        http://localhost:8000/docs
echo Health Check:    http://localhost:8000/health
echo.
echo Streamlit App:   http://localhost:8501
echo.
echo ========================================
echo Instructions:
echo ========================================
echo 1. Wait for both windows to fully load
echo 2. Open Streamlit in your browser (usually auto-opens)
echo 3. In Streamlit sidebar, check "Use FastAPI Backend"
echo 4. Verify API URL is: http://localhost:8000
echo.
echo To stop: Close the FastAPI and Streamlit windows
echo.
pause

