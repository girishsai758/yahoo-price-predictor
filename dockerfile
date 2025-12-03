# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# CRITICAL: Using official Python 3.12 Slim image to ensure version compliance.
# ----------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Set the working directory for dependency installation
WORKDIR /tmp/requirements

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file 
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir.
# This stage installs all necessary dependencies into the /usr/local/lib/python3.12/site-packages directory.
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# CRITICAL: Use the same lightweight Python 3.12 base for a clean, minimal runtime environment.
# ----------------------------------------------------------------------
FROM python:3.12-slim

# Set the application's main working directory
WORKDIR /app

# FIX: Copy installed site-packages from the builder stage.
# Since we are using python:3.12-slim, the standard site-packages path is correct!
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
# Ensure artifacts are present locally via DVC pull before the build
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl


# Expose the port for the Flask app
EXPOSE 5000

# FIX: Run Gunicorn using 'python -m' to ensure the executable is found in the container's environment.
# This fixes the "exec: gunicorn: executable file not found in $PATH" error.
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]