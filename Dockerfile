FROM python:3.11-slim

WORKDIR /app

# gcc and git are needed by mypy and some hypothesis backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Enable the openenv-core web interface (Gradio)
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
