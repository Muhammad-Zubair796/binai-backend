# 1. Use Ubuntu 24.04 (The only version with pre-compiled dlib)
FROM ubuntu:24.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python and the PRE-COMPILED dlib
# This avoids the 8GB RAM crash because we are NOT compiling anything.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dlib \
    python3-full \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Install face_recognition
# We use --break-system-packages because Ubuntu 24.04 is strict, 
# and --no-deps so it doesn't try to re-download dlib.
RUN pip3 install face-recognition-models --break-system-packages && \
    pip3 install face-recognition --no-deps --break-system-packages

# 6. Copy and install your other requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# 7. Copy your code
COPY . .

# 8. Start the app
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
