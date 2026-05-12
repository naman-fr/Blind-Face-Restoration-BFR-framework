# Multi-stage Dockerfile for BFR Framework
# Stage 1: Build environment
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for caching
COPY pyproject.toml .
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Stage 2: Runtime environment
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]
