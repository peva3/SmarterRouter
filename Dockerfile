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

RUN chmod +x /app/router/entrypoint.py

RUN useradd --create-home --shell /bin/bash router \
    && chown -R router:router /app \
    && mkdir -p /app/data \
    && chown router:router /app/data

# Note: We don't use USER here because we need to handle volume permissions at runtime
# The entrypoint will switch to the router user after setting up permissions

EXPOSE 11436

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:11436/health')" || exit 1

# Set entrypoint for automatic configuration
ENTRYPOINT ["python", "-m", "router.entrypoint"]
