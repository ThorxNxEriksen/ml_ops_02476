# Base image
FROM python:3.11-slim AS base

# Install build dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl gnupg && \
    echo "deb http://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt update && \
    apt install --no-install-recommends -y gcsfuse && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir --verbose
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy files
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY models models/
COPY reports reports/

# Install the application
RUN pip install . --no-deps --no-cache-dir --verbose

# Create a directory for mounting the GCS bucket
RUN mkdir /mnt/gcs-bucket

# Set the entry point
ENTRYPOINT ["/bin/bash", "-c", "gcsfuse --implicit-dirs quickdraw-databucket /mnt/gcs-bucket && python -u src/quick_draw/train_wandb.py --gcp-bucket --secret-manager"]