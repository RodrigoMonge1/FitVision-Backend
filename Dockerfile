FROM python:3.10-slim

# Instala dependencias del sistema necesarias para OpenCV y Mediapipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Crea directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto
COPY . /app

# Instala dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto 8080
EXPOSE 8080

# Comando para ejecutar tu app con gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
