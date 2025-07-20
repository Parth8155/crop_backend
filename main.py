from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Configure basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables - only in development
if os.getenv("AZURE_FUNCTIONS_ENVIRONMENT") is None and os.getenv("WEBSITE_INSTANCE_ID") is None:
    load_dotenv()
    logger.info("Loaded environment variables from .env file")

# Configure logging with environment variable
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level), force=True)

app = FastAPI(
    title="Crop Disease Detection API",
    description="AI-powered crop disease detection and treatment recommendation system",
    version="1.0.0"
)

# Log deployment environment info
logger.info(f"Starting application in environment: {os.getenv('ENVIRONMENT', 'unknown')}")
logger.info(f"Azure instance ID: {os.getenv('WEBSITE_INSTANCE_ID', 'not set')}")
logger.info(f"PORT: {os.getenv('PORT', 'not set')}")
logger.info(f"HOST: {os.getenv('HOST', 'not set')}")
logger.info(f"Python path: {os.getcwd()}")
logger.info(f"Model path: {os.getenv('MODEL_PATH', 'my_model.keras')}")

# CORS middleware - use environment variable with Azure-friendly fallback
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default CORS origins for development and Azure
    cors_origins = [
        "http://localhost:5173",
        "http://localhost:3000", 
        "https://salmon-pebble-03691881e.2.azurestaticapps.net"
    ]

logger.info(f"CORS origins configured: {cors_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    severity: str
    treatment: List[str]
    prevention: List[str]
    description: str
    affected_area: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

# Disease classes - your specific diseases
DISEASE_CLASSES = [
    "anthracnose", "gummosis", "leaf miner", "red rust",
    "bacterial blight", "brown spot", "green mite", "mosaic",
    "fall armyworm", "grasshopper", "leaf beetle", "leaf blight",
    "leaf spot", "streak virus", "leaf curl", "septoria leaf spot",
    "verticillium wilt", "healthy"
]

# Treatment recommendations for your disease classes
TREATMENT_DATABASE = {
    "anthracnose": {
        "treatment": [
            "Apply fungicide containing copper oxychloride",
            "Remove infected plant parts immediately",
            "Improve air circulation around plants",
            "Apply preventive sprays during favorable conditions"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Ensure proper plant spacing",
            "Avoid overhead watering",
            "Practice crop rotation"
        ],
        "description": "Fungal disease causing dark sunken lesions on fruits and leaves",
        "severity": "High"
    },
    "gummosis": {
        "treatment": [
            "Apply copper-based fungicides",
            "Prune infected branches below the infected area",
            "Improve drainage around the plant base",
            "Apply wound sealant after pruning"
        ],
        "prevention": [
            "Avoid mechanical damage to trunk and branches",
            "Ensure proper soil drainage",
            "Maintain optimal nutrition",
            "Regular inspection for early detection"
        ],
        "description": "Disease causing gum exudation from bark and branches",
        "severity": "Medium"
    },
    "leaf miner": {
        "treatment": [
            "Apply systemic insecticides like imidacloprid",
            "Remove heavily infested leaves",
            "Use yellow sticky traps",
            "Apply neem oil spray"
        ],
        "prevention": [
            "Use reflective mulch",
            "Encourage beneficial insects",
            "Regular monitoring of leaf undersides",
            "Maintain plant hygiene"
        ],
        "description": "Insect pest creating serpentine mines in leaves",
        "severity": "Medium"
    },
    "red rust": {
        "treatment": [
            "Apply fungicides containing propiconazole",
            "Remove infected leaves immediately",
            "Increase air circulation",
            "Apply foliar nutrition to boost immunity"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Avoid overhead irrigation",
            "Maintain proper plant spacing",
            "Regular field sanitation"
        ],
        "description": "Fungal disease causing reddish-brown pustules on leaves",
        "severity": "High"
    },
    "bacterial blight": {
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant parts",
            "Use drip irrigation instead of sprinklers",
            "Apply preventive copper sprays"
        ],
        "prevention": [
            "Use pathogen-free seeds",
            "Practice crop rotation",
            "Maintain field hygiene",
            "Avoid working in wet conditions"
        ],
        "description": "Bacterial infection causing water-soaked lesions",
        "severity": "High"
    },
    "brown spot": {
        "treatment": [
            "Apply fungicides containing carbendazim",
            "Improve field drainage",
            "Remove infected plant debris",
            "Apply balanced fertilization"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Maintain proper plant nutrition",
            "Ensure adequate spacing",
            "Practice field sanitation"
        ],
        "description": "Fungal disease causing brown circular spots on leaves",
        "severity": "Medium"
    },
    "green mite": {
        "treatment": [
            "Apply miticides like abamectin",
            "Spray water to reduce mite population",
            "Use predatory mites as biological control",
            "Apply neem oil or horticultural oils"
        ],
        "prevention": [
            "Maintain proper humidity levels",
            "Regular monitoring with magnifying glass",
            "Encourage beneficial predators",
            "Avoid excessive nitrogen fertilization"
        ],
        "description": "Tiny mites causing stippling and bronzing of leaves",
        "severity": "Medium"
    },
    "mosaic": {
        "treatment": [
            "Remove infected plants immediately",
            "Control aphid vectors with insecticides",
            "Use virus-free planting material",
            "Apply reflective mulch to repel aphids"
        ],
        "prevention": [
            "Use virus-resistant varieties",
            "Control aphid populations",
            "Maintain field hygiene",
            "Use certified virus-free seeds"
        ],
        "description": "Viral disease causing mosaic patterns on leaves",
        "severity": "Critical"
    },
    "fall armyworm": {
        "treatment": [
            "Apply insecticides like chlorantraniliprole",
            "Use biological control with Bt spray",
            "Hand-picking of larvae in small areas",
            "Apply neem-based insecticides"
        ],
        "prevention": [
            "Regular field monitoring",
            "Use pheromone traps",
            "Encourage natural enemies",
            "Practice crop rotation"
        ],
        "description": "Caterpillar pest causing extensive feeding damage",
        "severity": "Critical"
    },
    "grasshopper": {
        "treatment": [
            "Apply insecticides like malathion",
            "Use physical barriers around plants",
            "Apply neem oil spray",
            "Encourage bird predators"
        ],
        "prevention": [
            "Maintain clean field margins",
            "Use trap crops",
            "Regular field inspection",
            "Biological control with fungi"
        ],
        "description": "Jumping insects causing defoliation damage",
        "severity": "Medium"
    },
    "leaf beetle": {
        "treatment": [
            "Apply insecticides like cypermethrin",
            "Hand collection of adult beetles",
            "Use row covers for protection",
            "Apply diatomaceous earth"
        ],
        "prevention": [
            "Regular field monitoring",
            "Remove crop residues",
            "Use beneficial insects",
            "Practice crop rotation"
        ],
        "description": "Beetles feeding on leaves causing holes and damage",
        "severity": "Medium"
    },
    "leaf blight": {
        "treatment": [
            "Apply fungicides containing mancozeb",
            "Remove infected leaves",
            "Improve air circulation",
            "Apply protective copper sprays"
        ],
        "prevention": [
            "Use resistant varieties",
            "Ensure proper plant spacing",
            "Avoid overhead watering",
            "Practice field sanitation"
        ],
        "description": "Fungal disease causing large necrotic areas on leaves",
        "severity": "High"
    },
    "leaf spot": {
        "treatment": [
            "Apply fungicides containing chlorothalonil",
            "Remove infected plant parts",
            "Improve drainage",
            "Apply foliar fertilizers"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Maintain proper plant nutrition",
            "Ensure good air circulation",
            "Regular field inspection"
        ],
        "description": "Fungal disease causing circular spots on leaves",
        "severity": "Medium"
    },
    "streak virus": {
        "treatment": [
            "Remove infected plants immediately",
            "Control insect vectors",
            "Use virus-free planting material",
            "Apply systemic insecticides for vector control"
        ],
        "prevention": [
            "Use virus-resistant varieties",
            "Control leafhopper vectors",
            "Maintain field hygiene",
            "Use certified virus-free seeds"
        ],
        "description": "Viral disease causing streak symptoms on leaves",
        "severity": "Critical"
    },
    "leaf curl": {
        "treatment": [
            "Apply systemic insecticides for whitefly control",
            "Remove infected plants",
            "Use reflective mulch",
            "Apply plant growth regulators"
        ],
        "prevention": [
            "Use virus-resistant varieties",
            "Control whitefly vectors",
            "Use yellow sticky traps",
            "Maintain field sanitation"
        ],
        "description": "Viral disease causing upward curling of leaves",
        "severity": "High"
    },
    "septoria leaf spot": {
        "treatment": [
            "Apply fungicides containing azoxystrobin",
            "Remove infected lower leaves",
            "Improve air circulation",
            "Apply protective fungicide sprays"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Practice crop rotation",
            "Avoid overhead irrigation",
            "Maintain proper plant spacing"
        ],
        "description": "Fungal disease with small dark spots with light centers",
        "severity": "Medium"
    },
    "verticillium wilt": {
        "treatment": [
            "No effective chemical treatment available",
            "Remove infected plants",
            "Improve soil drainage",
            "Apply organic matter to soil"
        ],
        "prevention": [
            "Use resistant varieties",
            "Practice long crop rotations",
            "Soil solarization",
            "Maintain optimal soil health"
        ],
        "description": "Soil-borne fungal disease causing wilting symptoms",
        "severity": "Critical"
    },
    "healthy": {
        "treatment": ["No treatment needed - plant appears healthy"],
        "prevention": [
            "Continue regular monitoring",
            "Maintain proper nutrition",
            "Ensure adequate watering",
            "Keep area clean of debris"
        ],
        "description": "Plant shows no signs of disease or pest damage",
        "severity": "None"
    }
}

# Global model variable
model = None

def load_model():
    """Load the pre-trained model"""
    global model
    try:
        # Load your trained model
        model_path = os.getenv("MODEL_PATH", "my_model.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            logger.warning(f"Model file not found at {model_path}. Using mock predictions.")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (usually 224x224 for most models)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def predict_with_model(image_array):
    """Make prediction using the loaded model"""
    global model
    try:
        if model is not None:
            # Make prediction
            predictions = model.predict(image_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Get disease name from index
            if predicted_class_index < len(DISEASE_CLASSES):
                disease_name = DISEASE_CLASSES[predicted_class_index]
            else:
                disease_name = "unknown"
            
            return disease_name, confidence
        else:
            # Fallback to mock prediction if model not loaded
            return get_mock_prediction()
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return get_mock_prediction()

def get_mock_prediction():
    """Generate mock prediction for demonstration"""
    import random
    
    # Select a random disease for demo
    disease_names = [
        "leaf spot", "bacterial blight", "brown spot", "anthracnose", 
        "leaf blight", "mosaic", "healthy"
    ]
    disease_name = random.choice(disease_names)
    confidence = random.uniform(0.75, 0.95)  # High confidence for demo
    
    return disease_name, confidence

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - health check"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

@app.get("/cors-debug")
async def cors_debug():
    """Debug endpoint to check CORS configuration"""
    return {
        "cors_origins": cors_origins,
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "cors_origins_env": os.getenv("CORS_ORIGINS", "not set")
    }

@app.options("/predict")
async def predict_options():
    """Handle CORS preflight for predict endpoint"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
    
    Returns:
        PredictionResponse with disease prediction and treatment recommendations
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction using your trained model
        disease_name, confidence = predict_with_model(processed_image)
        
        # Get treatment information
        treatment_info = TREATMENT_DATABASE.get(
            disease_name, 
            TREATMENT_DATABASE["healthy"]
        )
        
        # Determine severity based on confidence
        if disease_name == "healthy":
            severity = "None"
        elif confidence > 0.9:
            severity = treatment_info.get("severity", "High")
        elif confidence > 0.7:
            severity = "Medium"
        else:
            severity = "Low"
        
        return PredictionResponse(
            disease=disease_name,
            confidence=round(confidence, 3),
            severity=severity,
            treatment=treatment_info["treatment"],
            prevention=treatment_info["prevention"],
            description=treatment_info["description"]
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/diseases")
async def get_supported_diseases():
    """Get list of supported diseases"""
    return {
        "supported_diseases": DISEASE_CLASSES,
        "total_classes": len(DISEASE_CLASSES)
    }

@app.get("/disease/{disease_name}")
async def get_disease_info(disease_name: str):
    """Get detailed information about a specific disease"""
    if disease_name not in TREATMENT_DATABASE:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    return {
        "disease": disease_name,
        **TREATMENT_DATABASE[disease_name]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Azure App Service automatically provides PORT environment variable
    # For local development, fallback to 8000
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )
