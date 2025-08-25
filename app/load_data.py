# ===============================
# load_data.py
# Этот файл будет забирать данные из Google Sheets и сохранять их в векторную базу Qdrant
# ===============================

import os
import pandas as pd                       # для работы с таблицами
import gspread                            # для работы с Google Sheets
from google.oauth2.service_account import Credentials  # для аутентификации через JSON-ключ
from qdrant_client import QdrantClient    # клиент для Qdrant
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer  # для создания эмбеддингов

# ===============================
# 1. Загружаем настройки
# ===============================

# В .env файле мы будем хранить путь к JSON-ключу и ID таблицы
from dotenv import load_dotenv
load_dotenv()

from app.config import (
    SERVICE_ACCOUNT_FILE,
    SPREADSHEET_ID,
    QDRANT_HOST,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM
)

# ===============================
# 2. Подключаемся к Google Sheets
# ===============================

# Указываем, какие права нужны — работать с Google Sheets и Google Drive
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", 
          "https://www.googleapis.com/auth/drive"]

# Создаем объект Credentials на основе JSON-ключа
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Авторизуемся через gspread
client = gspread.authorize(creds)

# Открываем таблицу по ID
spreadsheet = client.open_by_key(SPREADSHEET_ID)

# Берем первый лист (можно будет брать и по имени)
worksheet = spreadsheet.get_worksheet(0)

# Получаем все строки как список словарей
data = worksheet.get_all_records()

# Превращаем в pandas DataFrame (удобно для работы)
df = pd.DataFrame(data)
print("Данные загружены из Google Sheets:")
print(df.head())

# ===============================
# 3. Подключаемся к Qdrant
# ===============================

qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# Название коллекции
COLLECTION_NAME = "bowling_knowledge"

# Создаем коллекцию, если её нет
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)  # size=384 — это размерность эмбеддингов для модели all-MiniLM-L6-v2
    )
    print(f"Коллекция '{COLLECTION_NAME}' создана!")
else:
    print(f"Коллекция '{COLLECTION_NAME}' уже существует")

# ===============================
# 4. Создаем эмбеддинги
# ===============================

# Загружаем модель эмбеддингов
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Берем текст из колонки "text" (важно, чтобы в Google Sheets была колонка text!)
texts = df["text"].tolist()

# Превращаем тексты в вектора
embeddings = model.encode(texts)

# ===============================
# 5. Загружаем данные в Qdrant
# ===============================

# Готовим список для вставки
points = [
    PointStruct(
        id=i,                        # уникальный ID записи
        vector=embeddings[i],        # вектор эмбеддинга
        payload={"text": texts[i]}   # сам текст тоже сохраняем как payload
    )
    for i in range(len(texts))
]

# Загружаем в Qdrant
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Данные успешно загружены в Qdrant!")
