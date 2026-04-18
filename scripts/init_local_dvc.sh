#!/bin/bash
# pfa-mmcows/scripts/init_local_dvc.sh

SHARED_CACHE_DIR="/opt/dvc_shared_cache"

# Ensure we are inside the DVC repo
if [ ! -d ".dvc" ]; then
    echo "Error: Run this from the root of the pfa-mmcows repository."
    exit 1
fi

echo "Linking local repository to shared cache..."
dvc cache dir "$SHARED_CACHE_DIR"
dvc config cache.type symlink
dvc config cache.shared group

echo "Success! You can now safely run 'dvc pull'."