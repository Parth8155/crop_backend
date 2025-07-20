#!/bin/bash
# Azure startup script

echo "Starting Crop Disease Detection API on Azure..."

# Set environment variables if not already set
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

echo "Starting on $HOST:$PORT"

# Start the application with gunicorn
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind $HOST:$PORT --timeout 120
