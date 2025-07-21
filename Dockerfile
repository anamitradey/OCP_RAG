FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (only if needed; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# (Optional) pin pydantic v1 if using old validators
# In requirements.txt: pydantic<2

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Ensure the Chroma persistence dir exists & is writable (OpenShift random UID)
RUN mkdir -p /app/db && chmod -R g+rwX /app && chgrp -R 0 /app

EXPOSE 8080
# Start the server
CMD ["python", "app.py"]
# -----------------------------------------
