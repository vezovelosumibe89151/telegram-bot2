# 🚀 Развертывание Bowling RAG API на сервере

## 📋 Предварительные требования

### На сервере должно быть установлено:
- **Ubuntu/Debian** (рекомендуется Ubuntu 22.04+)
- **Docker** и **Docker Compose**
- **Git**
- **curl** для проверки здоровья

### Проверка установки:
```bash
# Docker
docker --version
docker-compose --version

# Git
git --version

# curl
curl --version
```

---

## 🛠️ Шаг 1: Подготовка сервера

### 1.1 Обновление системы
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Установка Docker
```bash
# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Установка Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Перезагрузка для применения группы docker
newgrp docker
```

### 1.3 Установка дополнительных инструментов
```bash
sudo apt install -y curl git htop ufw
```

### 1.4 Настройка firewall
```bash
# Разрешить SSH
sudo ufw allow ssh

# Разрешить HTTP и HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Включить firewall
sudo ufw enable
```

---

## 📁 Шаг 2: Клонирование и настройка проекта

### 2.1 Клонирование репозитория
```bash
cd ~
git clone https://github.com/vezovelosumibe89151/brooklyn-bowl-rag.git bowling-rag
cd bowling-rag
```

### 2.2 Настройка переменных окружения
```bash
# Копирование шаблона
cp .env.example app/.env

# Редактирование конфигурации
nano app/.env
```

**Заполните следующие переменные в `app/.env`:**

```bash
# GigaChat настройки
GIGACHAT_AUTH_KEY=ваш_gigachat_auth_key
GIGACHAT_BASE=https://api.gigachat.sber.ru
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat

# Qdrant настройки (Cloud)
QDRANT_URL=https://ваш_qdrant_url.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=ваш_qdrant_api_key
QDRANT_COLLECTION=bowling_knowledge

# Настройки приложения
TOP_K=5
MAX_QUERY_LEN=512
MAX_CONTEXT_LEN=2000
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2

# Секрет для Graph API
GRAPH_SECRET=ваш_секрет_для_graph_api

# Google Sheets (опционально)
SERVICE_ACCOUNT_FILE=service_account.json
SPREADSHEET_ID=ваш_google_sheets_id
```

### 2.3 Настройка Google Service Account (опционально)
```bash
# Загрузите service_account.json на сервер
# scp service_account.json user@server:~/bowling-rag/app/
```

---

## 🐳 Шаг 3: Запуск через Docker

### 3.1 Сборка и запуск
```bash
# Переход в директорию проекта
cd ~/bowling-rag

# Сборка образов
docker-compose build

# Запуск сервисов
docker-compose up -d
```

### 3.2 Проверка запуска
```bash
# Проверка состояния контейнеров
docker-compose ps

# Просмотр логов
docker-compose logs bowling-rag-api
```

### 3.3 Проверка здоровья API
```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# API документация
curl http://localhost:8000/docs
```

---

## 🌐 Шаг 4: Настройка веб-сервера (Nginx)

### 4.1 Установка Nginx
```bash
sudo apt install -y nginx
```

### 4.2 Настройка конфигурации
```bash
# Создание конфигурации
sudo nano /etc/nginx/sites-available/bowling-rag
```

**Содержимое файла `/etc/nginx/sites-available/bowling-rag`:**

```nginx
server {
    listen 80;
    server_name ваш_домен.com;

    # Проксирование запросов к Docker контейнеру
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Таймауты
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Логи
    access_log /var/log/nginx/bowling-rag_access.log;
    error_log /var/log/nginx/bowling-rag_error.log;
}
```

### 4.3 Активация сайта
```bash
# Создание символической ссылки
sudo ln -s /etc/nginx/sites-available/bowling-rag /etc/nginx/sites-enabled/

# Удаление дефолтной конфигурации
sudo rm /etc/nginx/sites-enabled/default

# Проверка конфигурации
sudo nginx -t

# Перезапуск Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## 🔒 Шаг 5: Настройка SSL (Let's Encrypt)

### 5.1 Установка Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 5.2 Получение SSL сертификата
```bash
sudo certbot --nginx -d ваш_домен.com
```

### 5.3 Автоматическое обновление сертификатов
```bash
# Проверка работы
sudo certbot renew --dry-run
```

---

## 📊 Шаг 6: Мониторинг и обслуживание

### 6.1 Просмотр логов
```bash
# Логи приложения
docker-compose logs -f bowling-rag-api

# Логи Nginx
sudo tail -f /var/log/nginx/bowling-rag_access.log
sudo tail -f /var/log/nginx/bowling-rag_error.log
```

### 6.2 Управление сервисом
```bash
# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Обновление
docker-compose pull && docker-compose up -d

# Просмотр использования ресурсов
docker stats
```

### 6.3 Резервное копирование
```bash
# Создание скрипта резервного копирования
cat > ~/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/$USER/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Резервная копия конфигурационных файлов
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    ~/bowling-rag/app/.env \
    ~/bowling-rag/app/service_account.json

echo "✅ Backup created: $BACKUP_DIR/config_$DATE.tar.gz"
EOF

chmod +x ~/backup.sh
```

---

## 🚨 Шаг 7: Устранение неполадок

### 7.1 Проблема: Контейнер не запускается
```bash
# Проверка логов
docker-compose logs bowling-rag-api

# Проверка конфигурации
docker-compose config
```

### 7.2 Проблема: API недоступен
```bash
# Проверка порта
netstat -tlnp | grep 8000

# Проверка из контейнера
docker-compose exec bowling-rag-api curl http://localhost:8000/health
```

### 7.3 Проблема: Недостаточно памяти
```bash
# Проверка использования памяти
docker stats

# Очистка неиспользуемых ресурсов
docker system prune -a
```

---

## 📋 Полезные команды

```bash
# Статус всех сервисов
docker-compose ps

# Просмотр логов в реальном времени
docker-compose logs -f

# Перезапуск конкретного сервиса
docker-compose restart bowling-rag-api

# Обновление и перезапуск
docker-compose pull && docker-compose up -d

# Остановка всех сервисов
docker-compose down

# Полная пересборка
docker-compose build --no-cache

# Проверка здоровья
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## 🎯 Следующие шаги

1. **Тестирование** - Проверьте все эндпоинты API
2. **Мониторинг** - Настройте логирование и алерты
3. **Безопасность** - Регулярно обновляйте сертификаты и пароли
4. **Масштабирование** - Настройте балансировку нагрузки при необходимости

**Ваш Bowling RAG API готов к работе! 🎉**
