FROM python:3.11-slim

LABEL org.opencontainers.image.title="SmarterRouter"
LABEL org.opencontainers.image.description="AI-powered LLM router that intelligently selects the best model"
LABEL org.opencontainers.image.url="https://github.com/peva3/SmarterRouter"
LABEL org.opencontainers.image.source="https://github.com/peva3/SmarterRouter"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash router \
    && chown -R router:router /app

USER router

EXPOSE 11436

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:11436/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11436"]
