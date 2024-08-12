FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install opencv-python-headless==4.8.0.76 \
    numpy==1.23.5 \
    mediapipe==0.10.0 \
    tensorflow-cpu==2.12.0 \
    uvicorn==0.22.0 \
    fastapi==0.95.0 \
    google-generativeai==0.7.2 \
    python-dotenv==1.0.1 \
    Jinja2==3.1.4 \
    python-multipart

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
