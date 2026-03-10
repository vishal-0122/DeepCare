# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Ensure models directory exists and copy model files
RUN mkdir -p models
COPY models/best_model.pkl models/best_model.pkl
COPY models/scaler.pkl models/scaler.pkl

# Expose ports
EXPOSE 8000 8501

# Start FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn app.serve:app --host 0.0.0.0 --port 8000 & streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0"]