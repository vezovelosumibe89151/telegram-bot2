# app/ingest_sheets.py

import uuid                                              # Для генерации уникальных ID точек (чанков)
from typing import List, Dict, Any                       # Подсказки типов — удобнее читать код
import numpy as np                                       # Работа с векторами (float32)
from tqdm import tqdm                                    # Красивый прогресс-бар
from datetime import datetime                            # Работа с датами (не обязательно, но полезно)

import gspread                                            # Клиент для Google Sheets
from google.oauth2.service_account import Credentials     # Аутентификация сервисного аккаунта

from sentence_transformers import SentenceTransformer     # Модель эмбеддингов (векторизация текста)
from qdrant_client import QdrantClient                    # Клиент Qdrant
from qdrant_client.models import VectorParams, Distance, PointStruct  # Описание коллекции/точек

import os
from dotenv import load_dotenv
load_dotenv()
from config import (
    SERVICE_ACCOUNT_FILE,
    SPREADSHEET_ID,
    QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    SHEETS_CONFIG,
    CHUNK_SIZE, CHUNK_OVERLAP
)

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
                "id": row.get("id"),
                "question": row.get("question"),
                "anwser": row.get("anwser"),
                "category": row.get("category"),
                "tags": row.get("tags"),
                "sourse": row.get("sourse"),
                "image_url": row.get("image_url"),
                "last_updated": row.get("last_updated")
            }
            anwser = payload.get("anwser", "")
            if not anwser:
                continue
            # Можно добавить чанкирование, если ответы длинные, иначе просто один чанк
            chunks = [anwser]
            for ch in chunks:
                vec = embedder.encode(ch).astype(np.float32)
                # Используем уникальный id из таблицы, если он есть, иначе генерируем UUID
                raw_id = payload.get("id")
                if raw_id is None or str(raw_id).strip() == "":
                    pid = str(uuid.uuid4())
                else:
                    pid = str(raw_id)
                point_payload = dict(payload)
                point_payload["anwser"] = ch
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
