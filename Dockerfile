# 1. Use Ubuntu 22.04 (Very stable, has pre-built ML packages)
FROM ubuntu:22.04

# 2. Prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python and the OFFICIAL PRE-COMPILED face-recognition packages
# This installs dlib and face_recognition as ready-to-use binaries.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-face-recognition \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy requirements (Make sure face_recognition, dlib, and numpy are NOT in here)
COPY requirements.txt .

# 6. Install your other requirements (FastAPI, Google Cloud, etc.)
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Copy your code
COPY . .

# 8. Start the app
# We use python3 -m uvicorn to ensure it uses the system-installed libraries
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
