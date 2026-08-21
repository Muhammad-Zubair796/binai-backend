# 1. Use Python 3.10 (The most stable version for face-recognition)
FROM python:3.10-slim

# 2. Install system dependencies and git
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install dlib from a PRE-COMPILED wheel (No 8GB crash)
RUN pip install --upgrade pip && \
    pip install https://github.com/jhelum-river/dlib-bin/raw/master/dlib-19.24.1-cp310-cp310-linux_x86_64.whl

# 4. Install face-recognition-models FIRST
RUN pip install face-recognition-models

# 5. Install face-recognition
RUN pip install face-recognition

# 6. Install your other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your code
COPY . .

# 8. Start the app
# We use uvicorn directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
