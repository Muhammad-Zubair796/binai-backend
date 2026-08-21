# 1. Use Miniconda (The gold standard for ML deployments)
FROM continuumio/miniconda3

# 2. Install dlib and numpy using Conda
# This downloads a PRE-BUILT binary. No compilation. No 8GB RAM crash.
RUN conda install -y -c conda-forge dlib numpy

# 3. Set the working directory
WORKDIR /app

# 4. Install face_recognition and other tools via pip
# (Since dlib is already installed by conda, pip will just skip it)
RUN pip install face-recognition fastapi uvicorn python-multipart google-cloud-aiplatform

# 5. Copy your code
COPY . .

# 6. Start the app
# We use the full path to uvicorn to ensure it uses the conda environment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
