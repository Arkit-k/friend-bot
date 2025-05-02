@echo off
echo Starting data collection and model training process...

REM Create necessary directories
mkdir data 2>nul
mkdir data\scraped 2>nul
mkdir data\huggingface 2>nul
mkdir data\processed 2>nul
mkdir data\reddit 2>nul
mkdir models 2>nul

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Run the data collection and training script
echo Running data collection and training script...
python collect_and_train.py %*

if %errorlevel% neq 0 (
    echo An error occurred during the process. Please check the logs for details.
    exit /b 1
)

echo Process completed successfully!
echo The trained model is saved as friendship_model.h5
echo The tokenizer is saved as tokenizer.pkl
echo The label encoder is saved as label_encoder.pkl

pause
