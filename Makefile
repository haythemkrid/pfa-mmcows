.PHONY: help init pull repro ui clean

# Default command when you just type 'make'
help:
	@echo "mmcows PFA Management Commands:"
	@echo "  make init      - Setup DVC and credentials"
	@echo "  make pull      - Pull latest code (Git) and data (DVC)"
	@echo "  make repro     - Run the entire DVC pipeline (dvc.yaml)"
	@echo "  make ui        - Start the MLflow UI on port 5000"
	@echo "  make clean     - Remove caches and temporary files"

# Initialize the project
init:
	chmod +x scripts/init.sh
	./scripts/init.sh

# Get everything up to date
pull:
	git pull
	dvc pull

# Run the MLOps pipeline defined in dvc.yaml
repro:
	dvc repro

# Launch MLflow to see experiment results
ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Cleanup
clean:
	rm -rf .dvc/tmp/cache
	find . -type d -name "__pycache__" -exec rm -rf {} +