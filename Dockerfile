FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8000
ENV OLLAYA_URL=http://127.0.0.1:11435
ENV OLLAYA_MCP_URL=http://127.0.0.1:11436

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
