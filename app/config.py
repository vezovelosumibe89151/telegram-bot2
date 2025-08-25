import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")              # <--- Укажите ваш Qdrant API ключ
COLLECTION_NAME = "brooklyn_cloud_faq"

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # <--- Укажите путь к вашему JSON-ключу Google Service Account
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")              # <--- Укажите ID вашей Google таблицы

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SHEETS_CONFIG = {
    "Rules": {
        "question_col": "Question",
        "answer_col": "Answer"
    },
    "History": {
        "question_col": "Event",
        "answer_col": "Description"
    }
}