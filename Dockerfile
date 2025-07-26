# Use slim Python base
FROM python:3.10-slim as base

# Install system dependencies (only what's needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Optional: Use pip cache mount for faster rebuilds (Docker 18.09+)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app ./app
COPY main.py .

# Optional: If using config, copy that too
COPY app/config.py ./app/config.py

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
