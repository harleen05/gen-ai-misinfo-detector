# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for image/audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
Flask==2.3.2
flask-cors==3.0.10
numpy==1.25.0
Pillow==10.0.0
librosa==0.10.1
tensorflow==2.14.0
scikit-learn==1.3.0
lime==0.2.2.1
gunicorn

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
deepfake-api/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ modeltext/
│   └─ checkpoints/
│       ├─ model.pkl
│       └─ vectorizer.pkl
├─ visual/
│   └─ vit_deepfake_visual.h5
├─ audio/
│   └─ checkpoints/
│       └─ best_model.pkl
└─ explanations_layer.py

# Set environment variable for Cloud Run port
ENV PORT 8080

# Expose port
EXPOSE 8080

# Start the Flask app with gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
