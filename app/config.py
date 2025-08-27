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
    "FAQ": {
        "id_col": "id",
        "question_col": "question",
        "answer_col": "anwser",
        "category_col": "category",
        "tags_col": "tags",
        "sourse_col": "sourse",
        "image_url_col": "image_url",
        "last_updated_col": "last_updated"
    }
}