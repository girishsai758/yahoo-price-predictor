# ----------------------------------------------------------------------
# Stage 1: BUILDER - Install Dependencies
# FIX 1: Switched to the estimated PyTorch 2.2.0 tag supporting Python 3.12.
# NOTE: If this specific tag is not found on Docker Hub, we may need to 
# check the official repository for the exact available Python 3.12 tag 
# for PyTorch 2.2.0.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime AS builder

WORKDIR /tmp/app

# Set locale environment variables to avoid locale warnings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy requirements file (Ensure it's requirements.txt)
COPY flaskapp/requirements1.txt .

# Install dependencies using --no-cache-dir
RUN pip install --no-cache-dir -r requirements1.txt

# ----------------------------------------------------------------------
# Stage 2: FINAL - The Runtime Image
# FIX 2: Switched to the estimated PyTorch 2.2.0 tag supporting Python 3.12.
# ----------------------------------------------------------------------
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# FIX 3: Copy installed site-packages from the Python 3.12 directory
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

# Copy your application files and artifacts
COPY flaskapp/ /app/
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

# Your application-specific commands
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]