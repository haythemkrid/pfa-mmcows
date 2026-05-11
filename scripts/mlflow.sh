#!/bin/bash

# 1. Clear the port
fuser -k 5000/tcp 2>/dev/null

# 2. Load variables from .env
if [ -f ~/pfa-mmcows/.env ]; then
    export $(grep -v '^#' ~/pfa-mmcows/.env | xargs)
    echo "Environment variables loaded from .env"
else
    echo "Error: .env file not found at ~/pfa-mmcows/.env"
    exit 1
fi

echo "Starting MLflow server with Remote Neon Postgres..."

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pfa-mmcows-env

# 3. Launch the server using the variable from .env
nohup python -m mlflow server \
    --backend-store-uri "$MLFLOW_DATABASE_URL" \
    --default-artifact-root /opt/mlflow_shared/artifacts \
    --host 127.0.0.1 \
    --port 5000 > /opt/mlflow_shared/mlflow.log 2>&1 &

echo "Done! MLflow is starting."