# Base image
FROM python:3.11-slim AS base

# Install build dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
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

RUN mkdir -p /data/processed
RUN mkdir -p /data/raw

# Install the application
RUN pip install . --no-deps --no-cache-dir --verbose

# Download and prepare data
RUN python src/quick_draw/data_download.py 
RUN python src/quick_draw/data.py 

# Set the entry point
ENTRYPOINT ["python", "-u", "src/quick_draw/train_wandb.py", "--no-gcp-bucket", "--secret-manager"]
