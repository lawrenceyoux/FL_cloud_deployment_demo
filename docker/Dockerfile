# docker/Dockerfile
# Single image used for ALL FL components:
#   Training job:  python -m src.train_baseline         (K8s Job)
#   FL server:     python -m src.federated.server       (K8s Deployment)
#   FL client:     python -m src.federated.client       (K8s Deployment)
#
# The CMD is overridden per K8s workload.
# All runtime config (MLflow URI, hospital ID, rounds, etc.) comes from
# env vars injected by the K8s ConfigMap / Job spec — no hardcoded values.
#
# Build locally:
#   docker build -t fl-stroke:latest .
#
# Run baseline training locally against a local MLflow:
#   docker run --rm \
#     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
#     fl-stroke:latest python -m src.train_baseline --epochs 30
#
# Build + push to ECR via the train-eks.yml GitHub Actions workflow.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure /app is always on sys.path so `from src.xxx import` works
# regardless of how the process is launched (python -m, uvicorn, etc.)
ENV PYTHONPATH=/app

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────────
# Copy requirements first so Docker cache layer survives source-only changes
COPY requirements.txt ./
# Install torch CPU-only wheel BEFORE requirements.txt so pip never resolves
# the default PyPI torch wheel which bundles ~2.5 GB of NVIDIA CUDA libraries.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Source code ──────────────────────────────────────────────────────────────
COPY src/ ./src/

# Write /app to the .pth file and smoke-test that src.models is discoverable.
# The Python check is written to a file first to avoid Dockerfile parser
# misinterpreting Python keywords (import, etc.) as Dockerfile instructions.
RUN printf '/app\n' > /usr/local/lib/python3.11/site-packages/fl_app.pth && \
    echo "sys.path:" && python -c "import sys; print(sys.path)" && \
    echo "src/ contents:" && ls /app/src/ && \
    echo "src/models/ contents:" && ls /app/src/models/ && \
    printf 'import importlib.util\nspec = importlib.util.find_spec("src.models")\nassert spec is not None, "FAIL: src.models not on sys.path"\nprint("OK: src.models found at", spec.origin)\n' > /tmp/check.py && \
    python /tmp/check.py

# ── Processed hospital data ──────────────────────────────────────────────────
# Bake the pre-processed CSVs into the image so the training job works
# without S3 access. The pipeline default path is local_dev/data/processed/.
# In production, set USE_S3=1 to pull fresh data from S3 instead.
COPY local_dev/data/processed/ ./local_dev/data/processed/

# ── Default command: baseline training ───────────────────────────────────────
# Overridden by K8s Job spec for server/client containers.
CMD ["python", "-m", "src.train_baseline"]
