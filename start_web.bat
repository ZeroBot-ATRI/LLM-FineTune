@echo off
echo Starting Qwen3-0.6B QLoRA Web Application...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM 激活虚拟环境（如果存在）
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM 检查依赖是否安装
echo Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM 启动Web应用
echo.
echo Starting web server...
echo Open your browser and go to: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

python start_web.py

pause