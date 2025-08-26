import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")              # <--- Укажите ваш Qdrant API ключ
COLLECTION_NAME = "bowling_knowledge"

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # <--- Укажите путь к вашему JSON-ключу Google Service Account
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")              # <--- Укажите ID вашей Google таблицы

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

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