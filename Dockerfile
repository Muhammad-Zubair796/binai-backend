# 1. Use the full Ubuntu 22.04 image
FROM ubuntu:22.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Enable the 'Universe' repository and install pre-compiled dlib
# This avoids the 8GB RAM crash because we are downloading a finished binary.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y \
    python3-pip \
    python3-dlib \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Install face_recognition
# We use --no-deps to prevent it from trying to download/compile dlib again
RUN pip3 install face-recognition-models && \
    pip3 install face-recognition --no-deps

# 6. Copy and install your other requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Copy your code
COPY . .

# 8. Start the app
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
