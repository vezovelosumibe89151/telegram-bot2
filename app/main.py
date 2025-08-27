from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from app.config import (
    QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME,
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
    api_key=QDRANT_API_KEY
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
        payload = point.payload
        results.append({
            "id": payload.get("id"),
            "question": payload.get("question"),
            "anwser": payload.get("anwser"),
            "category": payload.get("category"),
            "tags": payload.get("tags"),
            "sourse": payload.get("sourse"),
            "image_url": payload.get("image_url"),
            "last_updated": payload.get("last_updated"),
            "score": point.score
        })
    return {"results": results}

# Новый GET endpoint для поддержки запросов через URL
@app.get("/search")
async def search_get(query: str = Query(..., description="Поисковый запрос")):
    try:
        query_vec = embedder.encode(query).tolist()
        # Векторный поиск
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=20  # ищем больше кандидатов для keyword-фильтрации
        )
        # Keyword-фильтрация: ищем совпадение слова в question или anwser
        query_lower = query.lower()
        filtered = []
        for point in search_result:
            payload = point.payload
            question = str(payload.get("question", "")).lower()
            anwser = str(payload.get("anwser", "")).lower()
            if query_lower in question or query_lower in anwser:
                filtered.append(point)
        # Если keyword-совпадений 3 и больше — просим уточнить запрос
        if len(filtered) >= 3:
            return {"results": [], "message": "Опиши свой запрос более конкретно"}
        # Если есть keyword-совпадения — берем первый
        if filtered:
            point = filtered[0]
        else:
            point = search_result[0] if search_result else None
        results = []
        if point:
            payload = point.payload
            results.append({
                "id": payload.get("id"),
                "question": payload.get("question"),
                "anwser": payload.get("anwser"),
                "category": payload.get("category"),
                "tags": payload.get("tags"),
                "sourse": payload.get("sourse"),
                "image_url": payload.get("image_url"),
                "last_updated": payload.get("last_updated"),
                "score": point.score
            })
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}

# --- Комментарий: этот эндпоинт вызывается Gigachat Salutebot через Telegram-бота ---
# Пример запроса:
# {
#   "query": "Каковы правила игры в боулинг?",
#   "top_k": 3
# }
# Ответ: JSON с найденными результатами из вашей Google таблицы через Qdrant
