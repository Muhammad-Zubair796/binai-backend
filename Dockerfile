# 1. Use Debian Bookworm (This version has a pre-compiled dlib)
FROM debian:bookworm-slim

# 2. Install Python and the PRE-COMPILED dlib system package
# This takes 30 seconds and uses almost NO memory.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dlib \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Tell Python to look in the system folders for dlib
ENV PYTHONPATH=/usr/lib/python3/dist-packages

# 5. Install face_recognition WITHOUT its dependencies 
# (This prevents it from trying to download and compile dlib again)
RUN pip3 install face-recognition-models --break-system-packages
RUN pip3 install face-recognition --no-deps --break-system-packages

# 6. Install your other requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# 7. Copy your code
COPY . .

# 8. Start the app
# We use python3 -m uvicorn to ensure it uses the correct environment
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
