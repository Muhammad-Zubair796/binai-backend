# This image comes with Python 3.10, dlib, and face_recognition ALREADY installed.
# It bypasses the 8GB RAM build limit entirely.
FROM animenosekai/face_recognition:latest

# Set the working directory
WORKDIR /app

# Copy your requirements (Make sure face_recognition and dlib are NOT in here)
COPY requirements.txt .

# Install the other dependencies (FastAPI, Google Cloud, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Start the app
# We use uvicorn directly as it's already in the base image
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
