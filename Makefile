.PHONY: help init pull repro ui clean visual-index visual-yolo visual-train visual-pipeline

# Default command when you just type 'make'
help:
	@echo "mmcows PFA Management Commands:"
	@echo "  make init      - Setup DVC and credentials"
	@echo "  make pull      - Pull latest code (Git) and data (DVC)"
	@echo "  make repro     - Run the DVC pipeline"
	@echo "  make visual-index    - Build visual dataset index"
	@echo "  make visual-yolo     - Generate YOLO train/val/test + yaml"
	@echo "  make visual-train    - Train YOLOv8 model"
	@echo "  make visual-pipeline - Run visual index -> yolo -> train"
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

visual-index:
	python src/visual/data/a1_build_index.py --dataset-root store/data/raw/visual_data --date-folder 0725 --cameras cam_1,cam_2,cam_3,cam_4 --split-ratios 0.70,0.15,0.15 --seed 42

visual-yolo:
	python src/visual/data/a2_yolo_binary.py --dataset-root store/data/raw/visual_data

visual-train:
	python src/visual/models/train.py --data-yaml store/data/raw/visual_data/yolo_nano/mmcows_binary.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device cpu --workers 4 --project store/models/visual --name cow_detection_v8n

visual-pipeline:
	python src/visual/pipelines/training_pipeline.py --dataset-root store/data/raw/visual_data --date-folder 0725 --cameras cam_1,cam_2,cam_3,cam_4 --split-ratios 0.70,0.15,0.15 --seed 42 --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device cpu --workers 4 --run-name cow_detection_v8n

# Launch MLflow to see experiment results
ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Cleanup
clean:
	rm -rf .dvc/tmp/cache
	find . -type d -name "__pycache__" -exec rm -rf {} +