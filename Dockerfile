# 1. Use the official Miniconda image (This one worked for your build!)
FROM continuumio/miniconda3

# 2. Install git and system tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 3. Install dlib and numpy via Conda (Pre-built, no 8GB crash)
RUN conda install -y -c conda-forge dlib numpy

WORKDIR /app

# 4. Install face_recognition_models FIRST, then face-recognition
RUN pip install face-recognition-models
RUN pip install face-recognition
RUN pip install fastapi uvicorn python-multipart google-cloud-aiplatform

# 5. Copy your code
COPY . .

# 6. Start the app
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
