from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv

# -------------------------------
# 1. Загружаем переменные окружения из .env
# -------------------------------
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")       # URL Qdrant (например: https://xxx.qdrant.cloud)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # API ключ из облака
COLLECTION_NAME = "bowling_knowledge"

# -------------------------------
# 2. Подключаемся к Qdrant
# -------------------------------
client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)
# -------------------------------
# 3. Создаём коллекцию, если её ещё нет
# -------------------------------

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,                 # Размерность вектора (подходит для sentence-transformers)
            distance=Distance.COSINE  # Мера близости (косинусное расстояние)
        )
    )
    print(f" Коллекция '{COLLECTION_NAME}' создана!")
else:
    print(f" Коллекция '{COLLECTION_NAME}' уже существует")

# -------------------------------
# 4. Проверим список коллекций
# -------------------------------
print("Существующие коллекции:", client.get_collections())

