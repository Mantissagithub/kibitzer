# Kibitzer inference service

This service exposes the published `tactical_repair.pt` checkpoint as a small
CPU-only HTTP API. The client sends the game history as UCI moves. The service
rebuilds the board, runs PUCT, and returns Kibitzer's selected move plus its
root value and visit trace.

## Local container

Build from the repository root so the image can copy the `kibitzer` package:

```bash
docker build -f inference_service/Dockerfile -t kibitzer-inference .
docker run --rm -p 8080:8080 kibitzer-inference
```

Then request a move:

```bash
curl http://localhost:8080/move \
  -H 'content-type: application/json' \
  -d '{"moves":["e2e4"],"simulations":128}'
```

## Modal

Modal runs the same Dockerfile with one CPU, 1 GiB memory, one concurrent
request, and at most one container. It scales to zero after 60 idle seconds.

```bash
uvx modal setup
uvx modal deploy inference_service/modal_app.py
```

Set `VITE_KIBITZER_API_URL` in Vercel to the web function URL printed by the
deploy command. The production service currently uses this path.

## Cloud Run

The intended free-tier configuration is one CPU, 2 GiB memory, concurrency
one, no warm minimum, and one maximum instance. Build and push the container
from the repository root with Cloud Build:

```bash
gcloud builds submit \
  --config inference_service/cloudbuild.yaml \
  --substitutions _IMAGE=IMAGE_URL \
  .
```

Then deploy it with:

```bash
gcloud run deploy kibitzer-inference \
  --image IMAGE_URL \
  --region us-central1 \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated \
  --set-env-vars KIBITZER_ALLOWED_ORIGINS=https://kibitzer.vercel.app
```

Set `VITE_KIBITZER_API_URL` in the website build to the resulting service URL.
Do not add a browser-visible API secret. The container pins the published model
revision so a future Hub upload cannot silently change play.
