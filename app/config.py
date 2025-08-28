"""
Configuration file for Bowling RAG API
Loads environment variables and defines all application settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===========================================
# DATABASE CONFIGURATION
# ===========================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bowling_knowledge")

# ===========================================
# GOOGLE SHEETS CONFIGURATION
# ===========================================

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

# Sheet configuration with corrected column names
SHEETS_CONFIG = {
    "FAQ": {
        "id_col": "id",
        "question_col": "question",
        "answer_col": "answer",  # Fixed: was "anwser"
        "category_col": "category",
        "tags_col": "tags",
        "source_col": "source",  # Fixed: was "sourse"
        "image_url_col": "image_url",
        "last_updated_col": "last_updated",
        "uuid_col": "uuid"  # Added for data uniqueness
    }
}

# ===========================================
# EMBEDDING CONFIGURATION
# ===========================================

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# ===========================================
# TEXT PROCESSING CONFIGURATION
# ===========================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ===========================================
# GIGACHAT CONFIGURATION
# ===========================================

GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_BASE = os.getenv("GIGACHAT_BASE", "https://api.gigachat.sber.ru")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat:latest")

# ===========================================
# SBER GRAPH CONFIGURATION
# ===========================================

GRAPH_SECRET = os.getenv("GRAPH_SECRET")

# ===========================================
# RAG PARAMETERS
# ===========================================

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "1000"))
MAX_CONTEXT_LEN = int(os.getenv("MAX_CONTEXT_LEN", "4000"))

# ===========================================
# LOGGING CONFIGURATION
# ===========================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ===========================================
# SECURITY CONFIGURATION
# ===========================================

# CORS settings
ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]

# ===========================================
# DEVELOPMENT SETTINGS
# ===========================================

RELOAD = os.getenv("RELOAD", "false").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ===========================================
# VALIDATION FUNCTIONS
# ===========================================

def validate_config():
    """Validate that all required configuration is present"""
    required_vars = []

    if not QDRANT_HOST:
        required_vars.append("QDRANT_HOST")

    if not GIGACHAT_AUTH_KEY:
        required_vars.append("GIGACHAT_AUTH_KEY")

    if not GRAPH_SECRET:
        required_vars.append("GRAPH_SECRET")

    if required_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")

    return True

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        "database": {
            "qdrant_host": QDRANT_HOST,
            "collection_name": COLLECTION_NAME,
            "has_api_key": bool(QDRANT_API_KEY)
        },
        "google_sheets": {
            "has_service_account": bool(SERVICE_ACCOUNT_FILE),
            "spreadsheet_id": SPREADSHEET_ID[:10] + "..." if SPREADSHEET_ID else None
        },
        "gigachat": {
            "base_url": GIGACHAT_BASE,
            "model": GIGACHAT_MODEL,
            "has_auth_key": bool(GIGACHAT_AUTH_KEY)
        },
        "graph": {
            "has_secret": bool(GRAPH_SECRET)
        },
        "rag": {
            "top_k": TOP_K,
            "max_query_len": MAX_QUERY_LEN,
            "max_context_len": MAX_CONTEXT_LEN
        }
    }

# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your .env file and ensure all required variables are set.")