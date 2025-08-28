"""
Load data from Google Sheets to Qdrant
Simple data loading script for Bowling RAG system
"""

import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.config import (
    SERVICE_ACCOUNT_FILE,
    SPREADSHEET_ID,
    QDRANT_HOST,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    SHEETS_CONFIG
)

def authenticate_gsheets():
    """Authenticate with Google Sheets"""
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    print("✅ Google Sheets authentication successful")
    return client

def load_sheet_to_dataframe(gs_client, sheet_name: str) -> pd.DataFrame:
    """Load Google Sheet data into pandas DataFrame"""
    try:
        spreadsheet = gs_client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)

        # Get all records as dictionary
        records = worksheet.get_all_records()

        if not records:
            print(f"⚠️  Sheet '{sheet_name}' is empty")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        print(f"✅ Loaded {len(df)} rows from sheet '{sheet_name}'")
        return df

    except Exception as e:
        print(f"❌ Failed to load sheet '{sheet_name}': {e}")
        raise

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the DataFrame"""
    if df.empty:
        return df

    # Rename columns to match our config (fix typos)
    column_mapping = {
        'anwser': 'answer',  # Fix typo
        'sourse': 'source'   # Fix typo
    }

    df = df.rename(columns=column_mapping)

    # Add UUID column if not present
    if 'uuid' not in df.columns:
        df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
        print("✅ Added UUID column")

    # Fill missing values
    df = df.fillna('')

    # Ensure required columns exist
    required_cols = ['question', 'answer']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''

    # Filter out rows without content
    df = df[df['question'].str.len() > 0]

    print(f"✅ Processed DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df

def create_qdrant_points(df: pd.DataFrame, embedder) -> list:
    """Create Qdrant points from DataFrame"""
    points = []

    for idx, row in df.iterrows():
        try:
            # Extract data
            question = str(row.get('question', '')).strip()
            answer = str(row.get('answer', '')).strip()  # Fixed column name

            # Skip if no content
            if not question and not answer:
                continue

            # Create searchable text
            text_content = f"{question} {answer}".strip()

            # Generate embedding
            embedding = embedder.encode(text_content).tolist()

            # Prepare payload
            payload = {
                'uuid': str(row.get('uuid', str(uuid.uuid4()))),
                'question': question,
                'answer': answer,
                'category': str(row.get('category', '')),
                'tags': str(row.get('tags', '')),
                'source': str(row.get('source', '')),  # Fixed column name
                'image_url': str(row.get('image_url', '')),
                'last_updated': str(row.get('last_updated', '')),
                'created_at': datetime.now().isoformat()
            }

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )

            points.append(point)

        except Exception as e:
            print(f"❌ Error processing row {idx}: {e}")
            continue

    print(f"✅ Created {len(points)} Qdrant points")
    return points

def upload_to_qdrant(points: list):
    """Upload points to Qdrant"""
    if not points:
        print("⚠️  No points to upload")
        return

    try:
        # Initialize Qdrant client
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

        # Create collection if it doesn't exist
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print(f"✅ Created collection '{COLLECTION_NAME}'")
        else:
            print(f"ℹ️  Collection '{COLLECTION_NAME}' already exists")

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)

        print(f"✅ Successfully uploaded {len(points)} points to Qdrant")

    except Exception as e:
        print(f"❌ Failed to upload to Qdrant: {e}")
        raise

def main():
    """Main function"""
    print("🚀 Loading data from Google Sheets to Qdrant...")
    print("=" * 50)

    try:
        # Authenticate with Google Sheets
        print("🔐 Authenticating with Google Sheets...")
        gs_client = authenticate_gsheets()

        # Load data
        print("📊 Loading data from Google Sheets...")
        df = load_sheet_to_dataframe(gs_client, "FAQ")

        if df.empty:
            print("❌ No data found in Google Sheets")
            return

        # Process data
        print("🔄 Processing data...")
        df_processed = process_dataframe(df)

        # Initialize embedder
        print("🤖 Initializing embedder...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Embedder initialized: {EMBEDDING_MODEL_NAME}")

        # Create points
        print("📦 Creating Qdrant points...")
        points = create_qdrant_points(df_processed, embedder)

        # Upload to Qdrant
        print("☁️  Uploading to Qdrant...")
        upload_to_qdrant(points)

        print("=" * 50)
        print("🎉 Data loading completed successfully!")
        print(f"📊 Total records processed: {len(df_processed)}")
        print(f"📦 Total points uploaded: {len(points)}")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

if __name__ == "__main__":
    main()

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


# Берем текст из колонки "anwser"
texts = df["anwser"].tolist()

# Превращаем тексты в вектора
embeddings = model.encode(texts)

# ===============================
# 5. Загружаем данные в Qdrant
# ===============================


# Готовим список для вставки, payload содержит все поля строки
points = [
    PointStruct(
        id=row["id"],
        vector=embeddings[i],
        payload={
            "question": row["question"],
            "anwser": row["anwser"],
            "category": row["category"],
            "tags": row["tags"],
            "sourse": row["sourse"],
            "image_url": row["image_url"],
            "last_updated": row["last_updated"]
        }
    )
    for i, row in df.iterrows()
]

# Загружаем в Qdrant
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Данные успешно загружены в Qdrant!")
