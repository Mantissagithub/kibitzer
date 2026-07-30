from __future__ import annotations

import sys

import modal


app = modal.App("kibitzer-inference")
image = modal.Image.from_dockerfile("inference_service/Dockerfile")


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=300,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    sys.path.insert(0, "/app")
    from app import app as fastapi_app

    return fastapi_app
