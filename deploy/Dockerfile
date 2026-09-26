# sqljev gateway with Laya in-process. Serves SQL Server, Snowflake (SPCS), BigQuery (Cloud Run), Redshift.
#   docker build -f deploy/Dockerfile -t sqljev .                                   # CPU
#   docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t sqljev:gpu .
#   docker run -p 8765:8765 -e SQLJEV_GATEWAY_TOKEN=... sqljev
FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG MODEL=convaiinnovations/laya
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models PYTHONUNBUFFERED=1
RUN pip install torch --index-url ${TORCH_INDEX}
COPY pyproject.toml README.md LICENSE NOTICE /src/
COPY src /src/src
RUN pip install "/src[laya]"
# Bake the checkpoint into the image so the gateway starts without network access.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"
ENV SQLJEV_BACKEND=local HF_HUB_OFFLINE=1
EXPOSE 8765
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"
CMD ["sqljev", "gateway", "--host", "0.0.0.0", "--port", "8765"]
