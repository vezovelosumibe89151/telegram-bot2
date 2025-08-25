from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from app.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

# --- Класс запроса от Gigachat Salutebot ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# --- Инициализация FastAPI ---
app = FastAPI()

# --- Инициализация эмбеддера и Qdrant ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
    port=int(QDRANT_PORT) if QDRANT_PORT else None
)


# POST endpoint (старый)
@app.post("/search")
async def search(request: SearchRequest):
    query_vec = embedder.encode(request.query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=request.top_k
    )
    results = []
    for point in search_result:
        results.append({
            "doc_id": point.payload.get("doc_id"),
            "title": point.payload.get("title"),
            "text": point.payload.get("text"),
            "url": point.payload.get("url"),
            "image_url": point.payload.get("image_url"),
            "score": point.score
        })
    return {"results": results}

# Новый GET endpoint для поддержки запросов через URL
@app.get("/search")
async def search_get(query: str = Query(..., description="Поисковый запрос"), top_k: int = Query(3, description="Количество результатов")):
    query_vec = embedder.encode(query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )
    results = []
    for point in search_result:
        results.append({
            "doc_id": point.payload.get("doc_id"),
            "title": point.payload.get("title"),
            "text": point.payload.get("text"),
            "url": point.payload.get("url"),
            "image_url": point.payload.get("image_url"),
            "score": point.score
        })
    return {"results": results}

# --- Комментарий: этот эндпоинт вызывается Gigachat Salutebot через Telegram-бота ---
# Пример запроса:
# {
#   "query": "Каковы правила игры в боулинг?",
#   "top_k": 3
# }
# Ответ: JSON с найденными результатами из вашей Google таблицы через Qdrant
