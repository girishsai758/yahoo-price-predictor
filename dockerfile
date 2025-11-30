# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# USE OFFICIAL PYTORCH IMAGE to avoid massive installation in CI
# We switch to python:3.10-slim to python:3.10-cuda12.1-cudnn8-runtime
# This drastically reduces temporary disk space requirements.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /tmp/app

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy your requirements file
# NOTE: Ensure you use 'requirements.txt' or 'requirements1.txt' as appropriate.
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir
# Since we are using the PyTorch image, we only install the *rest* of the dependencies.
# NOTE: The dependency list includes 'gunicorn' and other tools needed for the final app.
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# This stage only copies the essential application code and site-packages.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy installed site-packages from the builder stage
# This transfers only the necessary Python dependencies installed in Stage 1.
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# Set the entry point for your Flask application
ENTRYPOINT ["/usr/local/bin/gunicorn"]
CMD ["--bind", "0.0.0.0:5000", "app:app"]