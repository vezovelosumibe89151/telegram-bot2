import os
from dotenv import load_dotenv

load_dotenv()

# ...существующие параметры...

# Qdrant
COLLECTION_NAME = "bowling_knowledge"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", QDRANT_URL)  # Для совместимости
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TOP_K = int(os.getenv("TOP_K", "5"))

# GigaChat
GIGACHAT_BASE = os.getenv("GIGACHAT_BASE", "https://api.gigachat.sber.ru")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

# Ограничения
MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "512"))
MAX_CONTEXT_LEN = int(os.getenv("MAX_CONTEXT_LEN", "2000"))
GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")

# Embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# Graph
GRAPH_SECRET = os.getenv("GRAPH_SECRET")  # Секрет для аутентификации от Sber Graph

# Google Sheets configuration
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # Путь к JSON-ключу Google Service Account
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")              # ID вашей Google таблицы

# Embedding and chunking
EMBEDDING_DIM = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Sheets configuration with corrected column names
SHEETS_CONFIG = {
    "FAQ": {
        "id_col": "id",
        "question_col": "question",
        "answer_col": "answer",  # Исправлено: "anwser" -> "answer"
        "category_col": "category",
        "tags_col": "tags",
        "source_col": "source",  # Исправлено: "sourse" -> "source"
        "image_url_col": "image_url",
        "last_updated_col": "last_updated"
    }
}