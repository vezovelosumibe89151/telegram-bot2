import os
import logging
from dotenv import load_dotenv
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# --- Загрузка переменных окружения ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "http://localhost:8000/search")

# --- Логирование ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip() if update.message and update.message.text else ""
    if not user_text:
        await update.message.reply_text("Пожалуйста, введите текст запроса.")
        return
    try:
        logger.info(f"Запрос от пользователя: {user_text}")
        response = requests.post(SEARCH_API_URL, json={"query": user_text, "top_k": 3}, timeout=10)
        if response.ok:
            results = response.json().get("results", [])
            if not results:
                await update.message.reply_text("Ничего не найдено.")
                return
            for r in results:
                reply = f"🏷 {r.get('title', '')}\n{r.get('text', '')}\n"
                if r.get('url'):
                    reply += f"🔗 {r['url']}\n"
                await update.message.reply_text(reply)
                # Отправка изображения, если есть image_url
                if r.get('image_url'):
                    try:
                        await update.message.reply_photo(r['image_url'])
                    except Exception as img_err:
                        logger.warning(f"Ошибка отправки изображения: {img_err}")
        else:
            logger.error(f"Ошибка поиска: {response.status_code} {response.text}")
            await update.message.reply_text("Ошибка поиска. Попробуйте позже.")
    except Exception as e:
        logger.exception(f"Ошибка обработки сообщения: {e}")
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не найден. Укажите токен в .env или переменных окружения.")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Telegram bot started.")
    app.run_polling()

if __name__ == "__main__":
    main()
