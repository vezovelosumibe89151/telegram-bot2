"""
Google Sheets data ingestion for Bowling RAG system
Loads data from Google Sheets, processes it, and stores in Qdrant vector database
"""

import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.config import (
    SERVICE_ACCOUNT_FILE,
    SPREADSHEET_ID,
    QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    SHEETS_CONFIG,
    CHUNK_SIZE, CHUNK_OVERLAP
)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks of specified size"""
    if not text:
        return []

    text = str(text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size]
        if len(piece) < 50:  # Skip very small chunks
            break
        chunks.append(piece)
        if start + chunk_size >= len(text):
            break

    return chunks

def ensure_collection(client: QdrantClient, vector_size: int):
    """Create or recreate collection if vector size changed"""
    if not client.collection_exists(COLLECTION_NAME):
        print(f"[INFO] Collection '{COLLECTION_NAME}' not found, creating new...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"[SUCCESS] Collection '{COLLECTION_NAME}' created")
    else:
        print(f"[INFO] Collection '{COLLECTION_NAME}' already exists")

def authenticate_gsheets():
    """Authenticate with Google Sheets API"""
    try:
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        client = gspread.authorize(creds)
        print("[SUCCESS] Google Sheets authentication successful")
        return client
    except Exception as e:
        print(f"[ERROR] Google Sheets authentication failed: {e}")
        raise

def load_sheet_data(gs_client: gspread.Client, sheet_name: str) -> List[Dict[str, Any]]:
    """Load data from specified Google Sheet"""
    try:
        spreadsheet = gs_client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)

        # Get all values
        all_values = worksheet.get_all_values()

        if not all_values:
            print(f"[WARNING] Sheet '{sheet_name}' is empty")
            return []

        # First row is headers
        headers = all_values[0]
        rows = all_values[1:]

        print(f"[INFO] Loaded {len(rows)} rows from sheet '{sheet_name}'")

        # Convert to dictionary format
        config = SHEETS_CONFIG.get(sheet_name, {})
        data = []

        for row_idx, row in enumerate(rows):
            if not any(row):  # Skip empty rows
                continue

            row_data = {}
            for col_idx, value in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx].lower().strip()
                    row_data[header] = value

            # Add UUID if not present
            if 'uuid' not in row_data or not row_data['uuid']:
                row_data['uuid'] = str(uuid.uuid4())

            # Validate required fields
            if not row_data.get('question') or not row_data.get('answer'):
                print(f"[WARNING] Row {row_idx + 2} missing question or answer, skipping")
                continue

            data.append(row_data)

        print(f"[SUCCESS] Processed {len(data)} valid entries from '{sheet_name}'")
        return data

    except Exception as e:
        print(f"[ERROR] Failed to load data from sheet '{sheet_name}': {e}")
        raise

def process_faq_data(faq_data: List[Dict[str, Any]]) -> List[PointStruct]:
    """Process FAQ data into Qdrant points"""
    points = []

    for item in tqdm(faq_data, desc="Processing FAQ data"):
        try:
            # Extract fields with corrected column names
            question = str(item.get('question', '')).strip()
            answer = str(item.get('answer', '')).strip()  # Fixed: was 'anwser'
            category = str(item.get('category', '')).strip()
            tags = str(item.get('tags', '')).strip()
            source = str(item.get('source', '')).strip()  # Fixed: was 'sourse'
            image_url = str(item.get('image_url', '')).strip()
            last_updated = str(item.get('last_updated', '')).strip()
            item_uuid = str(item.get('uuid', str(uuid.uuid4())))

            # Skip if no content
            if not question and not answer:
                continue

            # Create searchable text
            searchable_text = f"{question} {answer}".strip()

            # Create chunks if text is long
            if len(searchable_text) > CHUNK_SIZE:
                chunks = chunk_text(searchable_text, CHUNK_SIZE, CHUNK_OVERLAP)
            else:
                chunks = [searchable_text]

            # Create points for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for this chunk
                chunk_id = f"{item_uuid}_chunk_{chunk_idx}"

                # Prepare payload
                payload = {
                    "uuid": item_uuid,
                    "chunk_id": chunk_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "tags": tags,
                    "source": source,
                    "image_url": image_url,
                    "last_updated": last_updated,
                    "chunk_text": chunk,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "created_at": datetime.now().isoformat()
                }

                # Generate embedding for the chunk
                embedding = embedder.encode(chunk).tolist()

                # Create point
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

        except Exception as e:
            print(f"[ERROR] Failed to process item {item.get('uuid', 'unknown')}: {e}")
            continue

    print(f"[SUCCESS] Created {len(points)} points from {len(faq_data)} FAQ items")
    return points

def upload_to_qdrant(client: QdrantClient, points: List[PointStruct]):
    """Upload points to Qdrant in batches"""
    if not points:
        print("[WARNING] No points to upload")
        return

    batch_size = 100
    total_uploaded = 0

    try:
        # Clear existing data (optional - comment out if you want to keep existing data)
        print("[INFO] Clearing existing collection data...")
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        # Upload in batches
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_uploaded += len(batch)

        print(f"[SUCCESS] Uploaded {total_uploaded} points to Qdrant")

    except Exception as e:
        print(f"[ERROR] Failed to upload to Qdrant: {e}")
        raise

def main():
    """Main ingestion function"""
    print("🚀 Starting Bowling RAG data ingestion...")
    print("=" * 50)

    try:
        # Initialize embedder
        global embedder
        print("[INFO] Initializing embedder...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[SUCCESS] Embedder initialized: {EMBEDDING_MODEL_NAME}")

        # Initialize Qdrant client
        print("[INFO] Initializing Qdrant client...")
        qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        ensure_collection(qdrant_client, EMBEDDING_DIM)
        print("[SUCCESS] Qdrant client initialized")

        # Authenticate with Google Sheets
        print("[INFO] Authenticating with Google Sheets...")
        gs_client = authenticate_gsheets()

        # Load FAQ data
        print("[INFO] Loading FAQ data from Google Sheets...")
        faq_data = load_sheet_data(gs_client, "FAQ")

        if not faq_data:
            print("[ERROR] No data loaded from Google Sheets")
            return

        # Process data into points
        print("[INFO] Processing data into vector points...")
        points = process_faq_data(faq_data)

        # Upload to Qdrant
        print("[INFO] Uploading data to Qdrant...")
        upload_to_qdrant(qdrant_client, points)

        print("=" * 50)
        print("🎉 Data ingestion completed successfully!")
        print(f"📊 Total points uploaded: {len(points)}")
        print("🚀 Your RAG system is ready to use!")

    except Exception as e:
        print(f"[ERROR] Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Режем длинный текст на перекрывающиеся кусочки фиксированной длины."""
    if not text:                                         # Если пусто — возвращаем пустой список
        return []
    text = str(text).strip()                             # Страхуемся: приводим к строке и убираем пробелы по краям
    if len(text) <= chunk_size:                          # Если коротко — возвращаем как есть одним чанком
        return [text]
    chunks = []                                          # Список чанков
    step = max(1, chunk_size - overlap)                  # Шаг сдвига: длина чанка - перекрытие
    for start in range(0, len(text), step):              # Идём по тексту шагами
        piece = text[start:start + chunk_size]           # Берём подстроку нужного размера
        if len(piece) < 50:                              # Совсем крошечные кусочки не берём
            break
        chunks.append(piece)                             # Добавляем чанк в список
        if start + chunk_size >= len(text):              # Если это был последний чанк — выходим
            break
    return chunks                                        # Возвращаем список чанков

def ensure_collection(client: QdrantClient, vector_size: int):
    """Создаём коллекцию в Qdrant Cloud, если её нет или пересоздаём при изменении размера вектора."""
    if not client.collection_exists(COLLECTION_NAME):
        print(f"[INFO] Коллекция не найдена, создаём новую '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        print(f"[INFO] Коллекция '{COLLECTION_NAME}' уже существует.")

def get_gspread_client() -> gspread.Client:
    """Авторизуемся в Google с помощью сервисного аккаунта и получаем клиент gspread."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=scopes
    )
    return gspread.authorize(creds)

# --- Функция для получения прямой ссылки на файл Google Drive ---
def get_drive_image_url(file_id: str) -> str:
    # TODO: Вставьте свой Google Drive API ключ/логин, если требуется доступ к приватным файлам
    # Для публичных файлов достаточно такого формата:
    return f"https://drive.google.com/uc?id={file_id}"

def normalize_row(row: Dict[str, Any], mapping: Dict[str, Any], sheet_name: str) -> Dict[str, Any]:
    """
    Приводим строку листа к «унифицированному» виду: title/text/url/image_url/category/subcategory/...
    row — словарь вида {ИмяКолонки: Значение}; mapping — правила для текущего листа; sheet_name — имя листа.
    """
    # Достаём названия колонок из mapping (где что лежит)
    title_col = mapping.get("title_col")                  # Колонка с заголовком
    text_cols = mapping.get("text_cols", [])              # Список колонок, которые склеим в основной текст
    url_col = mapping.get("url_col")                      # Колонка со ссылкой
    image_col = mapping.get("image_col")                  # Колонка с фото (URL)
    branch_col = mapping.get("branch_col")                # Колонка с филиалом
    updated_col = mapping.get("updated_col")              # Колонка с датой обновления
    price_col = mapping.get("price_col")                  # Колонка с ценой
    allergens_col = mapping.get("allergens_col")          # Колонка с аллергенами
    subcategory_col = mapping.get("subcategory_col")      # Колонка с подкатегорией (может быть None)

    # Собираем текст из нескольких колонок (пустые игнорируем)
    text_parts = []                                       # Сюда соберём кусочки текста
    for col in text_cols:                                 # Идём по колонкам, указанным для текста
        val = row.get(col)                                # Берём значение ячейки
        if val and str(val).strip():                      # Если непустое — добавляем
            text_parts.append(str(val).strip())

    # Склеиваем в один текст (через перевод строки)
    main_text = "\n".join(text_parts) if text_parts else ""  # Если пусто — пустая строка

    # Формируем «унифицированную» запись (payload)
    payload = {
        "doc_id": f"{sheet_name}:{row.get(title_col, '')}",  # doc_id: имя листа + заголовок (простая стратегия)
        "category": mapping.get("category", sheet_name),      # Категория: из mapping или сам sheet_name
        "subcategory": row.get(subcategory_col, "") if subcategory_col else "",  # Подкатегория (если есть)
        "title": str(row.get(title_col, "")),                 # Заголовок
        "text": main_text,                                    # Основной текст (склейка колонок)
        "url": str(row.get(url_col, "")) if url_col else "", # Ссылка
        "image_url": "",                                     # Фото (URL)
        "branch_id": str(row.get(branch_col, "")) if branch_col else "",# Филиал
        "updated_at": str(row.get(updated_col, "")) if updated_col else "",  # Дата обновления
        "price": str(row.get(price_col, "")) if price_col else "",           # Цена
        "allergens": str(row.get(allergens_col, "")) if allergens_col else ""# Аллергены
    }
    if image_col and row.get(image_col):
        val = str(row.get(image_col, "")).strip()
        if val.startswith("http"):
            payload["image_url"] = val
        else:  # Если это file_id из Google Drive
            payload["image_url"] = get_drive_image_url(val)
    return payload                                         # Возвращаем нормализованные поля

def main():
    # Инициализируем эмбеддер (при первом запуске модель скачается)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)  # Загружаем модель эмбеддингов

    # Подключаемся к Qdrant
    client = QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )  # Создаём клиент Qdrant

    # Гарантируем, что коллекция существует и имеет правильную размерность
    vector_size = embedder.get_sentence_embedding_dimension()  # Получаем размерность эмбеддинга
    ensure_collection(client, vector_size)                     # Создаём коллекцию, если нет

    # Логинимся в Google и открываем таблицу
    gc = get_gspread_client()                                  # Авторизуемся gspread-клиентом
    sh = gc.open_by_key(SPREADSHEET_ID)                         # Открываем таблицу по ID

    points_batch = []                                          # Сюда будем складывать чанки для батч-вставки
    total_chunks = 0                                           # Счётчик созданных чанков

    # Идём по каждому листу согласно нашей карте SHEETS_MAPPING
    for sheet_name, mapping in SHEETS_CONFIG.items():         # Перебираем пары «имя листа» → «правила»
        try:
            ws = sh.worksheet(sheet_name)                      # Открываем конкретный лист по имени
        except Exception as e:
            print(f"[WARN] Лист '{sheet_name}' не найден: {e}")# Если листа нет — предупреждаем и продолжаем
            continue

        rows = ws.get_all_records()                            # Получаем все строки как список словарей {Колонка: Значение}
        if not rows:                                           # Если лист пустой
            print(f"[INFO] Лист '{sheet_name}' пуст.")
            continue

        for row in tqdm(rows, desc=f"Читаем лист '{sheet_name}'"):  # Прогресс по строкам листа
            # Формируем payload из всех нужных колонок FAQ
            payload = {
                "question": row.get("question"),
                "anwser": row.get("anwser"),
                "category": row.get("category"),
                "tags": row.get("tags"),
                "sourse": row.get("sourse"),
                "image_url": row.get("image_url"),
                "last_updated": row.get("last_updated")
            }
            question = str(payload.get("question") or "")
            anwser = str(payload.get("anwser") or "")
            if not anwser and not question:
                continue
            # Вектор строим по объединению question + anwser
            text_for_embedding = (question + " " + anwser).strip()
            if not text_for_embedding:
                continue
            vec = embedder.encode(text_for_embedding).astype(np.float32)
            # Используем уникальный id из таблицы, если он есть, иначе генерируем UUID
            # Всегда используем UUID для id
            pid = str(uuid.uuid4())
            point_payload = dict(payload)
            point = PointStruct(
                id=pid,
                vector=vec,
                payload=point_payload
            )
            points_batch.append(point)
            total_chunks += 1
            if len(points_batch) >= 500:
                print("POINTS TO UPSERT:", points_batch)
                client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
                points_batch = []

    # Записываем «хвост» (остатки менее 500)
    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)  # Финальная запись
    print(f"[INFO] Загружено {total_chunks} чанков в Qdrant Cloud.")
    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)  # Финальная запись

    print(f"[DONE] Загружено чанков: {total_chunks}")           # Отчёт о количестве загруженных чанков

if __name__ == "__main__":
    main()                                                      # Точка входа
