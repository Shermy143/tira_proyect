FROM python:3.9-slim

WORKDIR /app
# ---- Dependencias del sistema (mínimas) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencias Python ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# ---- Código fuente y modelo ----
COPY data_loader.py .
COPY train.py      .
COPY run.py        .
COPY ./models      /app/models

# ---- Variables de entorno ----
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# ---- Punto de entrada TIRA (shell form para expandir $inputDataset y $outputDir) ----
ENTRYPOINT ["python3", "/app/run.py", "-i", "$inputDataset", "-o", "$outputDir"]
