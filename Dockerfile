# Use a Python image with uv pre-installed
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy Python application code first
COPY data ./data
COPY models ./models
COPY modules ./modules
COPY utils ./utils
COPY app.py ./app.py

# Copy requirements files
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN ["uv", "pip", "install", "--system", "-r", "./requirements.txt"]

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]