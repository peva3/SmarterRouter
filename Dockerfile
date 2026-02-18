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

EXPOSE 11436

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11436"]
