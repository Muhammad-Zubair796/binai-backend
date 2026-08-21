FROM python:3.10-slim

# Install only the bare minimum system libraries
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install a PRE-COMPILED dlib wheel (This avoids the 8GB RAM crash)
RUN pip install https://github.com/jhelum-river/dlib-bin/raw/main/dlib-19.24.1-cp310-cp310-linux_x86_64.whl

# 2. Install face_recognition (It will now see dlib is already installed and won't compile)
RUN pip install face_recognition

# 3. Install the rest of your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
