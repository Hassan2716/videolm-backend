FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    ffmpeg tesseract-ocr tesseract-ocr-eng \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 curl git build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads outputs exports faiss_indexes logs
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
