FROM python:3.12-slim

WORKDIR /app

# Pre-install build tools so pip doesn't need to download them in isolated environments
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY . .

# Install dependencies with increased timeout, retries, and without redundant build isolation
RUN pip install --no-cache-dir --no-build-isolation --default-timeout=100 --retries=5 -e .

EXPOSE 5000