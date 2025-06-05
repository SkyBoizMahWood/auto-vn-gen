FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code selectively (excluding neo4j folder)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY api.py .
COPY main.py .

# Create logs directory
RUN mkdir -p logs

# Expose the API port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 