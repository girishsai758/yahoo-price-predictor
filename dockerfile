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

COPY flaskapp/ /app/

COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# After:
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

#local
CMD ["python", "app.py"]  

# #Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]