# Use the Python version that already works locally
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy runtime requirements and install
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY . .

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Command to run both the API and the UI
CMD ["sh", "-c", "python -m uvicorn src.prediction:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false"]