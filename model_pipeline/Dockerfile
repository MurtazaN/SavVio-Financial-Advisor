# =============================================================================
# SavVio ML Pipeline — Compute Environment
# =============================================================================
# This Dockerfile builds the container where model training code executes.
# By default it uses a lightweight Python image (CPU-only).
# To enable GPU/CUDA support for large datasets, swap the FROM line below.
# =============================================================================

# --------------------------------------------------------------
# Option 1: CPU-only (Default — lightweight, fast builds)
# --------------------------------------------------------------
FROM python:3.11-slim

# --------------------------------------------------------------
# Option 2: GPU/CUDA (Uncomment below AND comment out Option 1)
# Requires: NVIDIA Container Toolkit on host machine
# --------------------------------------------------------------
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y \
#     python3 python3-pip python3-dev \
#     build-essential cmake gcc g++ \
#     && rm -rf /var/lib/apt/lists/*
# --------------------------------------------------------------

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Copy pyproject.toml so pip recognizes this as a package
COPY pyproject.toml .

# Default: open a shell (runtime command is defined in docker-compose.yml)
CMD ["bash"]
