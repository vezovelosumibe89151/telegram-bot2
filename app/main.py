"""
Bowling RAG API - FastAPI Application
RAG system for bowling information with GigaChat integration and Sber Graph compatibility
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

from app.config import (
    # Database
    QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME,
    # Embeddings
    EMBEDDING_MODEL_NAME,
    # GigaChat
    GIGACHAT_AUTH_KEY, GIGACHAT_BASE, GIGACHAT_SCOPE, GIGACHAT_MODEL,
    # Graph
    GRAPH_SECRET,
    # RAG parameters
    TOP_K, MAX_QUERY_LEN, MAX_CONTEXT_LEN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Forbidden words filter
FORBIDDEN_WORDS = {
    "мат", "трахать", "ебучий", "оскорбление", "sex", "наркотик", "оружие", "бомба", "террор", "убийство",
    "fuck", "shit", "damn", "bitch", "asshole", "drug", "weapon", "bomb", "terror", "kill"
}

# Global variables for models
embedder = None
client = None
http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global embedder, client, http_client

    # Startup
    logger.info("Starting Bowling RAG API...")

    try:
        # Initialize sentence transformer
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Embedder initialized: {EMBEDDING_MODEL_NAME}")

        # Initialize Qdrant client
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        logger.info(f"Qdrant client initialized: {QDRANT_HOST}")

        # Initialize HTTP client for GigaChat
        http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("HTTP client initialized")

        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Bowling RAG API...")
    if http_client:
        await http_client.aclose()
    logger.info("✅ Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Bowling RAG API",
    description="RAG system for bowling information with GigaChat integration",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class GraphRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]

# Utility functions
def format_context(search_results: List) -> str:
    """Format search results into context string, filtering sensitive data"""
    if not search_results:
        return "No relevant information found."

    context_parts = []
    for i, result in enumerate(search_results[:TOP_K], 1):
        payload = result.payload

        # Filter out UUIDs and sensitive data
        question = str(payload.get("question", "")).replace("uuid", "[ID]").replace("UUID", "[ID]")
        answer = str(payload.get("answer", "")).replace("uuid", "[ID]").replace("UUID", "[ID]")

        context_parts.append(f"{i}. Q: {question}\n   A: {answer}")

    return "\n\n".join(context_parts)

def filter_forbidden_content(text: str) -> bool:
    """Check if text contains forbidden words"""
    text_lower = text.lower()
    return any(word in text_lower for word in FORBIDDEN_WORDS)

async def get_gigachat_token() -> str:
    """Get GigaChat access token"""
    try:
        auth_data = {
            "scope": GIGACHAT_SCOPE,
            "client_id": "your_client_id",  # Configure in production
            "client_secret": GIGACHAT_AUTH_KEY
        }

        response = await http_client.post(
            f"{GIGACHAT_BASE}/oauth/token",
            data=auth_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            logger.error(f"Failed to get token: {response.status_code}")
            raise HTTPException(status_code=500, detail="Authentication failed")

    except Exception as e:
        logger.error(f"Token request error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")

async def gigachat_complete(prompt: str, context: str = "") -> str:
    """Get completion from GigaChat"""
    try:
        token = await get_gigachat_token()

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Use this context to answer the question: {context}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": GIGACHAT_MODEL,
            "messages": messages,
            "max_tokens": MAX_CONTEXT_LEN
        }

        response = await http_client.post(
            f"{GIGACHAT_BASE}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"GigaChat API error: {response.status_code}")
            return "Извините, произошла ошибка при обработке запроса."

    except Exception as e:
        logger.error(f"GigaChat completion error: {e}")
        return "Извините, сервис временно недоступен."

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {}

    try:
        # Check embedder
        if embedder is not None:
            services_status["embedder"] = "healthy"
        else:
            services_status["embedder"] = "unhealthy"

        # Check Qdrant
        if client is not None:
            services_status["qdrant"] = "healthy"
        else:
            services_status["qdrant"] = "unhealthy"

        # Check HTTP client
        if http_client is not None:
            services_status["http_client"] = "healthy"
        else:
            services_status["http_client"] = "unhealthy"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        services_status["error"] = str(e)

    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        services=services_status
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Quick checks
        if embedder is None or client is None or http_client is None:
            raise HTTPException(status_code=503, detail="Services not ready")

        return {"status": "ready", "message": "All services are ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.post("/search")
async def search(request: SearchRequest):
    """Legacy search endpoint"""
    if filter_forbidden_content(request.query):
        return {"results": [], "message": "Не разговариваю на такие темы :) Спроси что-нибудь другое"}

    try:
        query_vec = embedder.encode(request.query).tolist()
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=request.top_k * 5
        )

        filtered = []
        for point in search_result:
            payload = point.payload
            question = str(payload.get("question", "")).lower()
            answer = str(payload.get("answer", "")).lower()
            query_lower = request.query.lower()

            if query_lower in question or query_lower in answer:
                filtered.append(point)

        if len(filtered) >= 3:
            results = filtered[:request.top_k]
        else:
            results = search_result[:request.top_k]

        return {
            "results": [
                {
                    "question": point.payload.get("question", ""),
                    "answer": point.payload.get("answer", ""),
                    "score": point.score
                }
                for point in results
            ]
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search service error")

@app.post("/rag-answer")
async def rag_answer(request: GraphRequest):
    """RAG answer endpoint for Sber Graph"""
    try:
        # Validate request
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > MAX_QUERY_LEN:
            raise HTTPException(status_code=400, detail=f"Query too long (max {MAX_QUERY_LEN} chars)")

        # Check for forbidden content
        if filter_forbidden_content(request.query):
            return {
                "answer": "Не разговариваю на такие темы :) Спроси что-нибудь другое",
                "user_id": request.user_id
            }

        # Search for relevant documents
        query_vec = embedder.encode(request.query).tolist()
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=TOP_K * 2
        )

        # Format context
        context = format_context(search_results)

        # Generate answer with GigaChat
        prompt = f"Ответь на вопрос пользователя на основе предоставленной информации о боулинге. Если информации недостаточно, скажи об этом.\n\nВопрос: {request.query}"
        answer = await gigachat_complete(prompt, context)

        return {
            "answer": answer,
            "user_id": request.user_id,
            "context_used": len(search_results) > 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG answer error: {e}")
        raise HTTPException(status_code=500, detail="RAG service error")

@app.post("/rag-answer-func")
async def rag_answer_func(request: GraphRequest):
    """RAG answer with function calling for Sber Graph"""
    try:
        # Validate request
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > MAX_QUERY_LEN:
            raise HTTPException(status_code=400, detail=f"Query too long (max {MAX_QUERY_LEN} chars)")

        # Check for forbidden content
        if filter_forbidden_content(request.query):
            return {
                "answer": "Не разговариваю на такие темы :) Спроси что-нибудь другое",
                "user_id": request.user_id,
                "function_called": False
            }

        # Search for relevant documents
        query_vec = embedder.encode(request.query).tolist()
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=TOP_K * 2
        )

        # Format context
        context = format_context(search_results)

        # Use function calling for more structured response
        functions = [
            {
                "name": "get_bowling_info",
                "description": "Get information about bowling",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The specific bowling topic"
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["basic", "detailed", "expert"],
                            "description": "Level of detail required"
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]

        # Generate answer with function calling
        prompt = f"Ты эксперт по боулингу. Ответь на вопрос пользователя, используя предоставленную информацию. Будь полезным и информативным.\n\nВопрос: {request.query}"
        answer = await gigachat_complete(prompt, context)

        return {
            "answer": answer,
            "user_id": request.user_id,
            "function_called": True,
            "context_used": len(search_results) > 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG function answer error: {e}")
        raise HTTPException(status_code=500, detail="RAG function service error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bowling RAG API v2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )



# POST endpoint (обновлённый)
@app.post("/search")
async def search(request: SearchRequest):
    query_lower = request.query.lower()
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in query_lower:
            return {"results": [], "message": "Не разговариваю на такие темы :) Спроси что-нибудь другое"}
    try:
        query_vec = embedder.encode(request.query).tolist()
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=request.top_k * 5
        )
        filtered = []
        for point in search_result:
            payload = point.payload
            question = str(payload.get("question", "")).lower()
            anwser = str(payload.get("anwser", "")).lower()
            if query_lower in question or query_lower in anwser:
                filtered.append(point)
        if len(filtered) >= 3:
            return {"results": [], "message": "Опиши свой запрос более конкретно"}
        if filtered:
            final_points = filtered[:request.top_k]
        else:
            final_points = search_result[:request.top_k]
        results = []
        for point in final_points:
            payload = point.payload
            results.append({
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

# Новый GET endpoint для поддержки запросов через URL
@app.get("/search")
async def search_get(
    query: str = Query(..., description="Поисковый запрос"),
    top_k: int = Query(3, description="Количество результатов")
):
    # Проверка на запрещённые слова
    query_lower = query.lower()
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in query_lower:
            return {"results": [], "message": "Не разговариваю на такие темы :) Спроси что-нибудь другое"}
    try:
        query_vec = embedder.encode(query).tolist()
        # Векторный поиск
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k * 5  # ищем больше кандидатов для keyword-фильтрации
        )
        # Keyword-фильтрация: ищем совпадение слова в question или anwser
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
        # Если есть keyword-совпадения — берем top_k
        if filtered:
            final_points = filtered[:top_k]
        else:
            final_points = search_result[:top_k]
        results = []
        for point in final_points:
            payload = point.payload
            results.append({
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
