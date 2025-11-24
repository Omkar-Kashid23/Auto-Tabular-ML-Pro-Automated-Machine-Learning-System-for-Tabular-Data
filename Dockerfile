# Use a stable Python base (3.10 recommended for binary compatibility)
FROM python:3.10-slim

# Create app dir
WORKDIR /app

# Avoid buffering issues
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system deps (for building wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose a port if you later add an API/streamlit (optional)
# EXPOSE 8501

# Default command: run the pipeline script
CMD ["python", "auto_ml.py"]

docker build -t auto-tabular-ml-pro .
docker run --rm -v "$(pwd)/full_automl_output:/app/full_automl_output" auto-tabular-ml-pro
