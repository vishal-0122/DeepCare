# Use stable Python slim image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY . .

# Ensure models directory exists
RUN mkdir -p models

# Expose ports
EXPOSE 8000 8501

# FastAPI = serve.py, Streamlit = app.py
CMD ["sh", "-c", "uvicorn app.serve:app --host 0.0.0.0 --port 8000 & streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0"]