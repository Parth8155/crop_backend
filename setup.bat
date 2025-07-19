@echo off

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create models directory
echo Creating models directory...
mkdir models

echo Setup complete!
echo To start the server, run:
echo venv\Scripts\activate
echo uvicorn main:app --reload --host 0.0.0.0 --port 8000
