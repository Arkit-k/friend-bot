@echo off
echo ===================================
echo Friendship Bot Startup
echo ===================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Create data directory if it doesn't exist
mkdir data 2>nul

REM Run the integration script
echo Running component integration...
python integrate_components.py

if %errorlevel% neq 0 (
    echo Integration failed. Please check the logs for details.
    pause
    exit /b 1
)

REM Check if model files exist
if not exist friendship_model.h5 (
    echo Model files not found. Creating dummy model files...
    python create_dummy_model.py
)

REM Start the bot
echo ===================================
echo Starting Friendship Bot...
echo ===================================
python discord_bot.py

if %errorlevel% neq 0 (
    echo An error occurred while running the bot. Please check the logs for details.
    pause
    exit /b 1
)

pause
