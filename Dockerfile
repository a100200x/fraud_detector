FROM python:3.12-slim

# Установка системных зависимостей (нужно, если библиотекам типа geopy или numpy понадобятся c-компиляторы)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Создаем структуру директорий сервиса
RUN mkdir -p /app/logs /app/input /app/output /app/train_data /app/models && \
    chmod -R 777 /app/logs /app/input /app/output

# 2. Ставим зависимости (этот слой кэшируется и не пересобирается при изменении кода)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Копируем исходный код проекта
COPY . .

# Переменная окружения, чтобы логи Python сразу выводились в консоль Docker, а не буферизировались
ENV PYTHONUNBUFFERED=1

# Исправленная точка запуска приложения (относительно WORKDIR /app)
CMD ["python", "app/app.py"]
