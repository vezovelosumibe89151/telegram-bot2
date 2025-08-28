import time
import base64
import httpx
import logging
import os
from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.config import (
    GIGACHAT_BASE, GIGACHAT_SCOPE, GIGACHAT_AUTH_KEY, GIGACHAT_MODEL,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, TOP_K, MAX_QUERY_LEN, MAX_CONTEXT_LEN,
    EMBEDDING_MODEL_NAME, COLLECTION_NAME, EMBEDDING_DIM
)
from dotenv import load_dotenv

# Загружаем переменные окружения из app/.env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# --- Логирование ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI(
    title="Bowling RAG API",
    description="RAG система для боулинга с интеграцией GigaChat и Sber Graph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint для мониторинга"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# --- Readiness check endpoint ---
@app.get("/ready")
async def readiness_check():
    """Readiness check - проверяет готовность всех зависимостей"""
    try:
        # Проверяем подключение к Qdrant
        if client is None:
            raise Exception("Qdrant client not initialized")

        # Проверяем доступность коллекции
        try:
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if COLLECTION_NAME not in collection_names:
                logger.warning(f"Collection '{COLLECTION_NAME}' not found. Available: {collection_names}")
                # Создаем коллекцию если не существует
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{COLLECTION_NAME}'")
        except Exception as qdrant_error:
            logger.error(f"Qdrant operation failed: {qdrant_error}")
            raise qdrant_error

        return {"status": "ready", "services": {"qdrant": "ok", "embedder": "ok"}}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(503, f"Service not ready: {str(e)}")

# --- Инициализация эмбеддера и Qdrant ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Инициализация Qdrant с fallback
try:
    if QDRANT_URL and QDRANT_URL.startswith('http'):
        logger.info(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        logger.info("Using local Qdrant instance")
        client = QdrantClient(host="localhost", port=6333)
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Qdrant client: {e}")
    logger.warning("Using mock client for development")
    client = None

# --- Кэш токена GigaChat ---
_token_cache = {"access_token": None, "expires_at": 0}

async def get_gigachat_token():
    # Попробуем использовать готовый токен из .env, если он есть
    access_token = os.getenv("GIGACHAT_ACCESS_TOKEN")
    if access_token:
        logger.info("Using access token from .env")
        return access_token
        
    now = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] - now > 30:
        logger.info("Using cached token")
        return _token_cache["access_token"]
    url = f"{GIGACHAT_BASE}/oauth/token"
    logger.info(f"Requesting token from: {url}")
    headers = {
        "Authorization": f"Basic {GIGACHAT_AUTH_KEY}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": GIGACHAT_SCOPE}
    logger.info(f"Auth headers: Authorization: Basic {GIGACHAT_AUTH_KEY[:20]}...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, data=data)
            logger.info(f"Token request status: {r.status_code}")
            if r.status_code != 200:
                logger.error(f"GigaChat auth failed: {r.status_code} {r.text}")
                raise HTTPException(500, f"Auth failed: {r.text}")
            tok = r.json()
            _token_cache["access_token"] = tok["access_token"]
            _token_cache["expires_at"] = now + int(tok.get("expires_in", 600))
            logger.info("Token obtained successfully")
            return _token_cache["access_token"]
    except httpx.RequestError as e:
        logger.error(f"Token request error: {e}")
        raise HTTPException(500, f"Token request error: {str(e)}")
    except Exception as e:
        logger.error(f"Token unexpected error: {e}")
        raise HTTPException(500, f"Token unexpected error: {str(e)}")

def format_context(points):
    chunks = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("answer") or payload.get("text") or payload.get("chunk") or ""
        question = payload.get("question") or payload.get("title") or ""
        url = payload.get("url") or payload.get("source") or payload.get("source_url") or ""
        tag = payload.get("tags") or payload.get("category") or ""
        # Исключаем id из текста — фильтруем url, если id случайно попал
        id_value = str(payload.get("id", ""))
        if id_value:
            url = url.replace(id_value, "").strip()
        piece = f"— {question} {f'[{tag}]' if tag else ''}\n{text}\nИсточник: {url}".strip()
        chunks.append(piece)
    context = "\n\n".join(chunks)
    # Ограничиваем длину контекста
    return context[:MAX_CONTEXT_LEN]

async def gigachat_complete(messages, attachments=None, stream=False, temperature=0.2, functions=None):
    token = await get_gigachat_token()
    logger.info(f"Token obtained: {token[:10]}...")
    url = f"{GIGACHAT_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "model": GIGACHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if attachments:
        body["attachments"] = attachments
    if stream:
        body["stream"] = True
    if functions:
        body["functions"] = functions
    logger.info(f"Sending request to GigaChat: {url}")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            logger.info(f"Attempting to connect to: {url}")
            r = await client.post(url, headers=headers, json=body)
            logger.info(f"GigaChat response status: {r.status_code}")
            if r.status_code != 200:
                logger.error(f"GigaChat error: {r.status_code} {r.text}")
                raise HTTPException(r.status_code, f"GigaChat error: {r.text}")
            return r.json()
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(500, f"Request error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, f"Unexpected error: {str(e)}")

# --- Функция для поиска документов (для function-calling) ---
def get_documents(query: str, top_k: int = TOP_K):
    """Функция для поиска релевантных документов в Qdrant."""
    if client is None:
        logger.warning("Qdrant client not available, returning empty results")
        return []

    try:
        q_vec = embedder.encode(query).tolist()
        res = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=qmodels.SearchParams(hnsw_ef=256, exact=False),
        )
        docs = []
        for p in res:
            payload = p.payload
            docs.append({
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "category": payload.get("category", ""),
                "tags": payload.get("tags", ""),
                "source": payload.get("source", ""),
                "score": p.score
            })
        return docs
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

# --- Модель запроса от Graph ---
class GraphRequest(BaseModel):
    userId: str
    queryText: str
    action: str = ""
    rawRequest: dict = {}

# --- Модель запроса для чата ---
class ChatRequest(BaseModel):
    message: str

# --- Endpoint /rag-answer для интеграции с Sber Graph ---
@app.post("/rag-answer")
async def rag_answer(req: Request, x_graph_secret: str = Header(None)):
    # Аутентификация
    graph_secret = os.getenv("GRAPH_SECRET")
    if not graph_secret or x_graph_secret != graph_secret:
        raise HTTPException(status_code=403, detail="Invalid Graph secret")

    # Проверка доступности Qdrant
    if client is None:
        return {"answer": "Сервис поиска временно недоступен. Попробуйте позже.", "buttons": []}

    data = await req.json()
    user_id = data.get("userId", "")
    query_text = data.get("queryText", "")
    action = data.get("action", "")
    raw_request = data.get("rawRequest", {})

    if not query_text:
        return {"answer": "Пожалуйста, задайте вопрос.", "buttons": []}

    # Ограничение длины запроса
    if len(query_text) > MAX_QUERY_LEN:
        return {"answer": f"Запрос слишком длинный (>{MAX_QUERY_LEN} символов).", "buttons": []}

    # Вариант A: Классический prompt-RAG
    # 1) Эмбеддинг вопроса
    q_vec = embedder.encode(query_text).tolist()
    # 2) Поиск в Qdrant
    try:
        res = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
            search_params=qmodels.SearchParams(hnsw_ef=256, exact=False),
        )
        hits = res
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return {"answer": "Ошибка поиска документов. Попробуйте позже.", "buttons": []}
    # 3) Контекст
    context = format_context(hits) if hits else ""
    # 4) Системный промпт
    system = (
        "Ты помощник для команды Brooklyn Bowl - сети боулинг ресторанов, отвечающий строго на основе предоставленного контекста.\n"
        "Ты доброжелателен, отвечаешь по делу и не уходишь в сторону.\n"
        "Не упоминай UUID, id или технические идентификаторы в ответе.\n"
        "Если в контексте нет ответа — скажи об этом прямо. Кратко, по делу.\n"
    )
    user_msg = f"Вопрос: {query_text}\n\nКонтекст:\n{context or '—'}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    try:
        completion = await gigachat_complete(messages)
        answer = completion["choices"][0]["message"]["content"]
        # Динамические кнопки (пример: если вопрос о правилах, предложить "Турниры")
        buttons = []
        if "правил" in query_text.lower():
            buttons = [{"text": "Расскажи о турнирах", "callback_data": "tournaments"}]
        return {"answer": answer, "buttons": buttons}
    except Exception as e:
        logger.exception(f"RAG error: {e}")
        return {"answer": f"Ошибка обработки запроса: {str(e)}", "buttons": []}

# --- Альтернативный endpoint с function-calling (Вариант B) ---
@app.post("/rag-answer-func")
async def rag_answer_func(req: Request, x_graph_secret: str = Header(None)):
    # Аутентификация
    graph_secret = os.getenv("GRAPH_SECRET")
    if not graph_secret or x_graph_secret != graph_secret:
        raise HTTPException(status_code=403, detail="Invalid Graph secret")

    data = await req.json()
    user_id = data.get("userId", "")
    query_text = data.get("queryText", "")

    if not query_text:
        return {"answer": "Пожалуйста, задайте вопрос.", "buttons": []}

    # Определение функций для GigaChat
    functions = [
        {
            "name": "get_documents",
            "description": "Поиск релевантных документов по запросу пользователя",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "top_k": {"type": "integer", "description": "Количество документов", "default": 5}
                },
                "required": ["query"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "Ты помощник по боулингу. Используй функцию get_documents для поиска информации. Не упоминай UUID, id или технические идентификаторы в ответе."},
        {"role": "user", "content": query_text}
    ]

    try:
        completion = await gigachat_complete(messages, functions=functions)
        message = completion["choices"][0]["message"]

        if "function_call" in message:
            # Модель вызвала функцию
            func_call = message["function_call"]
            if func_call["name"] == "get_documents":
                args = eval(func_call["arguments"])  # В продакшене использовать json.loads
                docs = get_documents(args["query"], args.get("top_k", TOP_K))
                # Добавляем результат функции в сообщения
                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": "get_documents",
                    "content": str(docs)
                })
                # Повторный вызов для финального ответа
                final_completion = await gigachat_complete(messages)
                answer = final_completion["choices"][0]["message"]["content"]
            else:
                answer = "Неизвестная функция."
        else:
            answer = message["content"]

        return {"answer": answer, "buttons": []}
    except Exception as e:
        logger.exception(f"Function-calling error: {e}")
        return {"answer": f"Ошибка: {str(e)}", "buttons": []}

# --- Старые endpoints для совместимости ---
class RAGRequest(BaseModel):
    query: str
    user_id: str | None = None
    chat_id: str | None = None

@app.post("/rag-answer-old")
async def rag_answer_old(req: RAGRequest):
    logger.info(f"RAG answer old called with query: {req.query}, user_id: {req.user_id}")
    # Перенаправление на новый endpoint (для совместимости)
    import json
    fake_request = {"userId": req.user_id or "", "queryText": req.query}
    fake_body = json.dumps(fake_request).encode('utf-8')
    
    async def fake_receive():
        return {"body": fake_body, "type": "http.request", "more_body": False}
    
    fake_request_obj = Request(
        scope={"type": "http", "method": "POST", "path": "/", "headers": []}, 
        receive=fake_receive
    )
    logger.info("Calling rag_answer with fake request")
    result = await rag_answer(fake_request_obj, x_graph_secret=os.getenv("GRAPH_SECRET"))
    logger.info(f"RAG answer old result: {result}")
    return result

# --- Endpoint для общения с GigaChat напрямую ---
@app.post("/chat")
async def chat_with_gigachat(request: ChatRequest):
    logger.info(f"Chat endpoint called with message: {request.message}")
    try:
        logger.info(f"Received chat request with message: {request.message}")
        messages = [{"role": "user", "content": request.message}]
        completion = await gigachat_complete(messages)
        answer = completion["choices"][0]["message"]["content"]
        logger.info(f"Chat response: {answer}")
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
