FROM python:3.10-slim

# Deps mínimas para OpenCV/Mediapipe en servidores
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala deps primero (mejor cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copia el código (incluye el .pkl)
COPY . .

ENV PORT=8080
EXPOSE 8080

# Gunicorn (ajusta workers/threads si tu instancia es pequeña)
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "2", "-b", "0.0.0.0:8080", "app:app"]
