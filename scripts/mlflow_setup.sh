#!/bin/bash

echo "1. Cleaning up old MLflow processes on port 5000..."
fuser -k 5000/tcp 2>/dev/null

echo "2. Setting up shared directory permissions..."
mkdir -p /opt/mlflow_shared/artifacts
chown -R root:mmcows-team /opt/mlflow_shared
# The 's' (setgid) ensures all new files created inside inherit the mmcows-team group
chmod -R 2775 /opt/mlflow_shared

echo "3. Loading Conda environment..."
# This line teaches the script how to use 'conda activate'
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pfa-mmcows-env

echo "4. Starting MLflow server in the background..."
nohup python -m mlflow server \
    --backend-store-uri sqlite:////opt/mlflow_shared/mlflow.db \
    --default-artifact-root /opt/mlflow_shared/artifacts \
    --host 127.0.0.1 \
    --port 5000 > /opt/mlflow_shared/mlflow.log 2>&1 &

echo "Done! MLflow is starting."
echo "You can check for errors by running: cat /opt/mlflow_shared/mlflow.log"
#!/bin/bash

# 1. Clear the port
sudo fuser -k 5000/tcp 2>/dev/null

# 2. Setup your Neon URL (I've added the 'ql' to postgresql)
# Replace the string below with your full string from the image if different
NEON_URL="postgresql://neondb_owner:npg_4jJ0LWOAPoCR@ep-silent-glade-anaturzt-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require"

echo "Starting MLflow server with Remote Neon Postgres..."

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pfa-mmcows-env

# 3. Launch the server
# Notice we keep --default-artifact-root local for now. 
# This means parameters go to the Cloud, but heavy model files stay on your disk.
nohup python -m mlflow server \
    --backend-store-uri "$NEON_URL" \
    --default-artifact-root /opt/mlflow_shared/artifacts \
    --host 127.0.0.1 \
    --port 5000 > /opt/mlflow_shared/mlflow.log 2>&1 &

echo "Done! Check logs at /opt/mlflow_shared/mlflow.log"