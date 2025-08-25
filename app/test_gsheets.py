import gspread
from google.oauth2.service_account import Credentials
from app.config import SERVICE_ACCOUNT_FILE, SPREADSHEET_ID

# Настраиваем доступ
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
client = gspread.authorize(creds)

# Пробуем открыть таблицу
sheet = client.open_by_key(SPREADSHEET_ID)
print("✅ Connected to:", sheet.title)

# Получаем первый лист
worksheet = sheet.get_worksheet(0)
print("📊 First row:", worksheet.row_values(1))