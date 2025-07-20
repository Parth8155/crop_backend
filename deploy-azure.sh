#!/bin/bash

echo "🚀 Deploying to Azure App Service..."

# Build and deploy backend
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🔧 Starting application..."
python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
