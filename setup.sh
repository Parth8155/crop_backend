#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows - use this line instead on Windows

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory
echo "Creating models directory..."
mkdir -p models

echo "Setup complete!"
echo "To start the server, run:"
echo "source venv/bin/activate"  # Linux/Mac
echo "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
