# Crop Disease Detection Backend

A FastAPI-based backend for AI-powered crop disease detection and treatment recommendations using your trained Keras model.

## Features

- **Custom Model Integration**: Uses your `my_model.keras` trained model
- **18 Disease Classes**: Supports detection of 18 specific crop diseases plus healthy classification
- **Real-time Predictions**: Fast image processing and disease prediction
- **Treatment Recommendations**: Detailed treatment and prevention guidelines for each disease
- **RESTful API**: Clean, documented API endpoints
- **CORS Support**: Ready for frontend integration

## Supported Diseases

The model can detect the following 18 disease conditions:

1. **anthracnose** - Fungal disease causing dark sunken lesions
2. **gummosis** - Disease causing gum exudation from bark
3. **leaf miner** - Insect pest creating serpentine mines in leaves  
4. **red rust** - Fungal disease with reddish-brown pustules
5. **bacterial blight** - Bacterial infection with water-soaked lesions
6. **brown spot** - Fungal disease with brown circular spots
7. **green mite** - Tiny mites causing leaf stippling
8. **mosaic** - Viral disease with mosaic patterns on leaves
9. **fall armyworm** - Caterpillar pest causing feeding damage
10. **grasshopper** - Jumping insects causing defoliation
11. **leaf beetle** - Beetles feeding on leaves
12. **leaf blight** - Fungal disease with large necrotic areas
13. **leaf spot** - Fungal disease with circular spots
14. **streak virus** - Viral disease causing streak symptoms
15. **leaf curl** - Viral disease causing upward leaf curling
16. **septoria leaf spot** - Fungal disease with small dark spots
17. **verticillium wilt** - Soil-borne fungal disease causing wilting
18. **healthy** - No disease detected

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Your trained model file: `my_model.keras`

### Installation (Windows)

1. **Place Your Model**: Copy your `my_model.keras` file to the backend directory

2. **Install Dependencies**: Double-click `install_dependencies.bat` or run:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Start the Server**: Double-click `start_server.py` or run:
   ```bash
   # Activate virtual environment
   venv\Scripts\activate
   
   # Start server
   python start_server.py
   ```

4. **Access API Documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`
   - Health Check: `http://localhost:8000/`

## API Endpoints

### Health Check
- `GET /` - API health status and model loading status

### Prediction
- `POST /predict` - Upload image and get disease prediction
  - **Input**: Multipart form data with image file
  - **Output**: Disease name, confidence, severity, treatment, and prevention recommendations

### Disease Information
- `GET /diseases` - List all supported diseases
- `GET /disease/{disease_name}` - Get detailed info about specific disease

## Usage Examples

### Predict Disease from Image
```python
import requests

# Upload image for prediction
url = "http://localhost:8000/predict"
files = {"file": open("crop_image.jpg", "rb")}
response = requests.post(url, files=files)

prediction = response.json()
print(f"Disease: {prediction['disease']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Treatment: {prediction['treatment']}")
```

### Get Disease Information
```python
import requests

# Get info about specific disease
url = "http://localhost:8000/disease/Tomato Late Blight"
response = requests.get(url)

disease_info = response.json()
print(disease_info)
```

## Model Integration

### Using Your Trained Model

1. **Place your model**: Copy `my_model.keras` to the backend directory
2. **Model will be automatically loaded** when the server starts
3. **If model not found**, the API will use mock predictions for testing

### Model Requirements

Your model should:
- Accept input images of shape `(224, 224, 3)` (RGB)
- Output predictions for 18 classes in this order:
  ```python
  [
      "anthracnose", "gummosis", "leaf miner", "red rust",
      "bacterial blight", "brown spot", "green mite", "mosaic",
      "fall armyworm", "grasshopper", "leaf beetle", "leaf blight",
      "leaf spot", "streak virus", "leaf curl", "septoria leaf spot",
      "verticillium wilt", "healthy"
  ]
  ```
- Return softmax probabilities (sum to 1.0)

## Production Deployment

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Environment Variables

Create a `.env` file for configuration:
```env
MODEL_PATH=models/crop_disease_model.h5
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp,tiff
LOG_LEVEL=INFO
```

## Response Format

### Prediction Response
```json
{
  "disease": "Tomato Late Blight",
  "confidence": 0.923,
  "severity": "Critical",
  "treatment": [
    "Apply fungicides containing chlorothalonil or copper",
    "Remove infected plants immediately",
    "Destroy infected tubers",
    "Improve field drainage"
  ],
  "prevention": [
    "Plant certified disease-free seed potatoes",
    "Ensure proper spacing for air circulation",
    "Avoid overhead irrigation during humid conditions",
    "Monitor weather conditions for favorable disease development"
  ],
  "description": "Devastating fungal disease that can destroy entire potato crops"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Successful prediction
- `400`: Invalid image format or bad request
- `404`: Disease not found
- `500`: Internal server error

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
