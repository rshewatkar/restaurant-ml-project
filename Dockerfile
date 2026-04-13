FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face uses 7860
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run correct app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]