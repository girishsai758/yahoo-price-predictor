# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# This stage uses --no-cache-dir to save temporary space and is discarded later.
# ----------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /tmp/app

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir to minimize temporary layer size
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# This stage only copies the essential application code and site-packages.
# ----------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy installed site-packages from the builder stage
# This is the line that transfers only the installed Python dependencies
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# Your application-specific commands
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 
# Assuming you use Gunicorn or similar WSGI server