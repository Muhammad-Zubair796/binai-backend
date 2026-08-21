# 1. Use the updated Anaconda image
FROM anaconda/miniconda3

# 2. Install git (needed to download the face models) and dlib
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN conda install -y -c conda-forge dlib numpy

WORKDIR /app

# 3. Install the specific models that the error message asked for
RUN pip install git+https://github.com/ageitgey/face_recognition_models

# 4. Install the rest of the libraries
RUN pip install face-recognition fastapi uvicorn python-multipart google-cloud-aiplatform

COPY . .

# 5. Start the app using 'python -m uvicorn' to ensure it finds the conda path
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
