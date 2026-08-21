# 1. Use a slim Python image
FROM python:3.10-slim

# 2. Install system dependencies required for dlib and face_recognition
# We combine these to keep the image small
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install Python dependencies
# CRITICAL: MAKEFLAGS="-j1" prevents the 8GB RAM crash by using only 1 CPU core for compilation
RUN MAKEFLAGS="-j1" pip install --no-cache-dir -r requirements.txt

# 6. Copy your main.py and any other files
COPY . .

# 7. Start the FastAPI app using uvicorn
# Render uses the PORT environment variable, usually 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
