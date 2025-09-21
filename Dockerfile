# Use Python slim image
FROM python:3.10-slim

# Prevent interactive prompts
ENV PYTHONUNBUFFERED True

# Set work directory
WORKDIR /app

# Install system dependencies (needed for librosa, PIL, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Set environment variables
ENV PORT 8080
ENV GEMINI_API_KEY="AIzaSyAImUuxo1t3jipw2IF0AY5FB5-N4y8enzg"

# Expose port
EXPOSE 8080

# Run Flask
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
