#!/usr/bin/env python3
"""
Start script for the Crop Disease Detection API
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Change to the backend directory
    os.chdir(script_dir)
    
    print("🌱 Starting Crop Disease Detection API...")
    print("📋 API Documentation will be available at:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("🔧 API Health Check: http://localhost:8000/")
    print()
    
    # Check if model file exists
    model_path = "my_model.keras"
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("   The API will use mock predictions for demonstration.")
    print()
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[script_dir],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
