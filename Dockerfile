FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PAPER_CASH=1000 \
    CRON_SECRET=dev-paper-secret

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY FINDINGS_ML.md FINDINGS.md FINDINGS_AGGRESSIVE.md ./
# seed JSON so the UI has charts before the first live sync
COPY dashboard/seed_state.json ./dashboard/seed_state.json

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Pre-seed empty data dirs (volume can mount over /app/data)
RUN mkdir -p /app/data /app/output

EXPOSE 8080
CMD ["uvicorn", "profit_engine.server:app", "--host", "0.0.0.0", "--port", "8080"]
