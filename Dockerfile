# Use TensorFlow base image so TF is already installed
FROM tensorflow/tensorflow:2.15.0

# Prevent interactive prompts
ENV PYTHONUNBUFFERED True

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt


# Copy app code
COPY . .

# Environment variables
ENV PORT 8080
ENV GEMINI_API_KEY=your-gemini-api-key-here

# Expose API port
EXPOSE 8080

# Run Flask app via Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
