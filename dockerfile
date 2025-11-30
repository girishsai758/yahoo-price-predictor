# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# USE OFFICIAL PYTORCH IMAGE to avoid massive installation in CI
# We switch to python:3.10-slim to python:3.10-cuda12.1-cudnn8-runtime
# This drastically reduces temporary disk space requirements.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime-py312 AS builder

WORKDIR /tmp/app

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file (Ensure it's requirements.txt)
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir
# Since we are using the PyTorch image, we only install the *rest* of the dependencies.
# NOTE: The dependency list includes 'gunicorn' and other tools needed for the final app.
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# This stage only copies the essential application code and site-packages.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime-py312

WORKDIR /app

# FIX 3: Copy installed site-packages from the Python 3.12 directory
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# Your application-specific commands
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]