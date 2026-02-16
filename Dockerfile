FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 11434

ENV ROUTER_OLLAMA_URL=http://host.docker.internal:11434

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11434"]
