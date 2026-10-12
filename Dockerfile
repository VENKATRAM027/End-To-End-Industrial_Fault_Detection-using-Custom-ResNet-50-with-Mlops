From python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    &&rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip intsall --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e

EXPOSE 8000 8501