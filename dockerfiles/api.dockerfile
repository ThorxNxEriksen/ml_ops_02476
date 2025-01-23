# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./src /src
COPY ./app /app
COPY ./models /models
COPY app/backend_requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uvicorn app.backend_api:app --host 0.0.0.0 --port ${PORT:-8000}"]

## Run the api using
# gcloud run deploy api \
#     --image=europe-west1-docker.pkg.dev/quickdrawproject-448508/quickdraw-artifacts/api_image:latest \
#     --region=europe-west1 \
#     --platform=managed \
#     --allow-unauthenticated \
#     --memory=1Gi


## Stop the api using
# gcloud run services delete api     --region=europe-west1     --platform=managed
