FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KIBITZER_CHECKPOINT=/model/tactical_repair.pt \
    KIBITZER_TORCH_THREADS=1 \
    PORT=8080

WORKDIR /app

COPY inference_service/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt chess==1.11.2

ARG MODEL_REVISION=678e50b52d49ca03e0b0d05b54d83796b2c25044
RUN mkdir -p /model \
    && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Pradheep1647/kibitzer-tactical-repair', filename='tactical_repair.pt', revision='${MODEL_REVISION}', local_dir='/model')"

COPY kibitzer /app/kibitzer
COPY inference_service/app.py /app/app.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
