# # ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# Using the specified PyTorch image for the build environment.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime AS builder

# Set the working directory for dependency installation
WORKDIR /tmp/requirements

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file (using the filename found in your last message)
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# Using the specified PyTorch image for the runtime environment.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# CRITICAL FIX: The PyTorch image likely uses Python 3.10, not 3.12.
# We are changing the path from python3.12 to python3.10.
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
# Ensure artifacts are present locally via DVC pull before the build
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl


# Set environment variable for Flask app (optional, but good practice)
ENV FLASK_APP=app.py

# Expose the port for the Flask app
EXPOSE 5000
# FIX: Run Gunicorn using 'python -m' to ensure the executable is found in the container's environment.
# This fixes the "exec: gunicorn: executable file not found in $PATH" error.
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# ----------------------------------------------------------------------
# FROM python:3.12-slim AS builder

# WORKDIR /tmp/requirements

# # Install setuptools FIRST, before other dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # Install setuptools and wheel first (required for building some packages)
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# # Copy and install requirements
# COPY flaskapp/requirements1.txt .
# RUN pip install --no-cache-dir -r requirements1.txt

# # ----------------------------------------------------------------------
# # Stage 2: FINAL - The Runtime Image
# # ----------------------------------------------------------------------
# FROM python:3.12-slim

# WORKDIR /app

# # Install minimal system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     && rm -rf /var/lib/apt/lists/*

# # Copy installed packages from builder
# COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
# COPY --from=builder /usr/local/bin/ /usr/local/bin/

# # Copy application files
# COPY flaskapp/ /app/
# COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# EXPOSE 5000

# CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]