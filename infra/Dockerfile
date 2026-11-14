FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY contracts/ ./contracts/
COPY collector/ ./collector/
COPY paper_trader/ ./paper_trader/
COPY engine/ ./engine/

RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1 \
    PAPER_ONLY=true

CMD ["python", "-m", "paper_trader.runner"]
