# 1. Use the Miniconda image (This one successfully passes the 8GB build limit)
FROM continuumio/miniconda3

# 2. Install git (Required to download the models from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 3. Install dlib and numpy via Conda (Pre-built, no 8GB crash)
RUN conda install -y -c conda-forge dlib numpy

# 4. Set the working directory
WORKDIR /app

# 5. Install the EXACT model package requested by the error log
RUN pip install git+https://github.com/ageitgey/face_recognition_models

# 6. Install the rest of the libraries
RUN pip install face-recognition fastapi uvicorn python-multipart google-cloud-aiplatform

# 7. Copy your code
COPY . .

# 8. Start the app
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
