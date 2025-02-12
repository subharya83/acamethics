FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for pdfplumber
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY genQA.py .
COPY fineTuneSLM.py .
COPY querySLM.py .

# Create directories for data, models, and weights
RUN mkdir -p /app/data /app/models /app/weights

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for CUDA (if using GPU)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set default command (can be overridden)
CMD ["python", "--version"]