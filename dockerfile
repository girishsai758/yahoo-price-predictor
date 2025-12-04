# # ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# Using the specified PyTorch image for the build environment.
# ----------------------------------------------------------------------
# FIX: Using the user-specified PyTorch 2.8.0 image, compatible with Python 3.12
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime AS builder

# Set the working directory for dependency installation
WORKDIR /tmp/requirements

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file (using the filename found in your last message)
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir
# NOTE ON WARNING: If your requirements file contains 'torch', 'torchvision', or 'torchaudio',
# this step will uninstall the pre-installed versions, which leads to a dependency conflict 
# warning. It is best practice to remove these core libraries from requirements1.txt if the base image provides them.
RUN pip install --no-cache-dir -r requirements1.txt \
    # DEBUG STEP: The packages are not in /usr/local/lib, check /opt/conda
    && echo "DEBUG: Installed packages path is likely /opt/conda/lib/python3.12/site-packages" 

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# Using the specified PyTorch image for the runtime environment.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# CRITICAL FIX 1: Change COPY path to the common Conda/PyTorch installation location.
# This should fix the "not found" error by targeting the correct site-packages folder.
COPY --from=builder /opt/conda/lib/python3.11/site-packages/ /opt/conda/lib/python3.11/site-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
# Ensure artifacts are present locally via DVC pull before the build
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl


# Set environment variable for Flask app (optional, but good practice)


# Expose the port for the Flask app
EXPOSE 5000

# CRITICAL FIX 2: Use Python module execution to ensure Gunicorn is always found 
# in the environment path, fixing potential "gunicorn not found" issues.
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