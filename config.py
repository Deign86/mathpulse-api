"""Configuration management for MathPulse AI Backend"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    
    # Hugging Face
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Model settings
    USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "False").lower() == "true"
    
    # Model IDs for Hugging Face Inference API
    # Using more capable models for better document understanding
    CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")  # Upgraded from 1.5B
    CLASSIFICATION_MODEL_ID = os.getenv("CLASSIFICATION_MODEL_ID", "facebook/bart-large-mnli")
    
    # Document extraction model (for unstructured text)
    EXTRACTION_MODEL_ID = os.getenv("EXTRACTION_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    
    # API endpoints (updated to new router.huggingface.co)
    HF_INFERENCE_API_URL = "https://router.huggingface.co/hf-inference/models"

config = Config()
