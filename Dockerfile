FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

WORKDIR /app
# ---- Dependencias del sistema (mínimas) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencias Python ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages && rm -rf /root/.cache/pip

# ---- Código fuente y modelo ----
COPY data_loader.py .
COPY train.py      .
COPY run.py        .

# ---- Variables de entorno ----
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_OFFLINE=1

# ---- Punto de entrada TIRA (shell form para expandir $inputDataset y $outputDir) ----
ENTRYPOINT ["python3", "/app/run.py", "--model_path", "/root/.cache/huggingface/hub/models--Doffy143--mStyleDistance-Finetunningv2/snapshots/6e61470abe58b6b5652881469dc480dc98700fee/", "-i", "$inputDataset", "-o", "$outputDir"]
