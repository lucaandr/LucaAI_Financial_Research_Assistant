# --- Base image --------------------------------------------------------------
FROM python:3.11-slim

# System build tools: chromadb (via hnswlib) needs a compiler if a prebuilt
# wheel isn't available for this platform; curl is used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies -------------------------------------------------------
# Install CPU-only torch FIRST, from PyTorch's own CPU wheel index. Without this,
# `pip install -r requirements.txt` resolves the default GPU build of torch (a
# dependency of sentence-transformers/transformers) on Linux, which drags in
# several GB of unused NVIDIA/CUDA packages (nvidia-cusparselt, nvidia-nccl,
# cuda-toolkit, triton, ...) — this alone is what made the build take an hour.
# We don't have/use a GPU inside the container, so none of that is needed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Now install everything else — torch's requirement is already satisfied by the
# CPU-only install above, so pip won't pull in the GPU version.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers embeddings model at BUILD time.
# Otherwise every container's first RAG query has to fetch it from the
# Hugging Face Hub at runtime (~4-5s + external network dependency + the
# rate-limit/latency issues we saw in the logs).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- Application code -----------------------------------------------------------
COPY . .

# Data/log folders the app writes to. Created here so the app doesn't error
# out on a fresh container before any file has been uploaded; in practice
# these should be mounted as volumes (see docker-compose.yml) so data
# survives container restarts.
RUN mkdir -p data/raw_pdfs data/chromadb logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --server.address=0.0.0.0 is required in Docker: by default Streamlit only
# listens on localhost inside the container, which is unreachable from
# outside it.
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]