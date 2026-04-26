.PHONY: help init pull repro ui clean visual-index visual-yolo visual-train visual-pipeline check-gpu mbt-run mbt-multi

# Visual pipeline defaults (override at runtime, e.g. make visual-train EPOCHS=10 DEVICE=0)
DATASET_ROOT ?= store/data/raw/visual_data
DATE_FOLDER ?= 0725
CAMERAS ?= cam_1,cam_2,cam_3,cam_4
SPLIT_RATIOS ?= 0.70,0.15,0.15
SEED ?= 42
MODEL ?= yolov8n.pt
EPOCHS ?= 100
IMGSZ ?= 640
BATCH ?= 16
DEVICE ?= 0
WORKERS ?= 4
RUN_NAME ?= cow_detection_v8n
VISUAL_EXPERIMENT_NAME ?= Visual_Training
REQUIRE_GPU ?= 1
DATA_YAML ?= $(DATASET_ROOT)/yolo_nano/mmcows_binary.yaml
VISUAL_PROJECT ?= store/models/visual

# Multimodal MBT runner defaults (Hydra overrides are passed through as plain args)
MBT_RUN_NAME ?= mbt_single_run
MBT_OVERRIDES ?=
MBT_EXPERIMENTS ?= mbt_baseline,mbt_s2,mbt_lr3e4
MBT_BASE_OVERRIDES ?=
MBT_EXPERIMENTS_FILE ?= configs/mbt_experiments.yaml
MBT_STOP_ON_ERROR ?= 1

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
	@echo "  make mbt-run         - Run one multimodal MBT experiment"
	@echo "  make mbt-multi       - Run multiple multimodal MBT experiments"
	@echo "  Example overrides:"
	@echo "    make visual-train EPOCHS=10 DEVICE=0 RUN_NAME=exp_gpu"
	@echo "    make visual-index DATE_FOLDER=0726 CAMERAS=cam_1,cam_3"
	@echo "    make visual-pipeline MODEL=yolov8s.pt BATCH=8 WORKERS=2 VISUAL_EXPERIMENT_NAME=Visual_Exp"
	@echo "    make visual-pipeline DEVICE=0 REQUIRE_GPU=1"
	@echo "    make visual-pipeline DEVICE=cpu REQUIRE_GPU=0"
	@echo "    make mbt-run MBT_RUN_NAME=mbt_test MBT_OVERRIDES=\"training.epochs=2 output.use_mlflow=false\""
	@echo "    make mbt-multi"
	@echo "    make mbt-multi MBT_EXPERIMENTS=mbt_a,mbt_b MBT_BASE_OVERRIDES=\"training.epochs=5\" MBT_EXPERIMENTS_FILE="
	@echo "    make mbt-multi MBT_EXPERIMENTS_FILE=configs/mbt_experiments.yaml"
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
	python -m src.visual.data.a1_build_index --dataset-root $(DATASET_ROOT) --date-folder $(DATE_FOLDER) --cameras $(CAMERAS) --split-ratios $(SPLIT_RATIOS) --seed $(SEED)

visual-yolo:
	python -m src.visual.data.a2_yolo_binary --dataset-root $(DATASET_ROOT) --remap-labels-in-place

check-gpu:
	@if [ "$(REQUIRE_GPU)" = "1" ]; then \
		if ! command -v nvidia-smi >/dev/null 2>&1; then \
			echo "[ERROR] GPU required but nvidia-smi not found. Set REQUIRE_GPU=0 to bypass."; \
			exit 1; \
		fi; \
		echo "[OK] NVIDIA GPU detected by nvidia-smi."; \
		python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 || { \
			echo "[ERROR] GPU required but torch.cuda.is_available() is False."; \
			echo "        nvidia-smi may work while PyTorch CUDA is unusable (often driver/runtime mismatch)."; \
			echo "        Fix NVIDIA driver/CUDA stack, or run with REQUIRE_GPU=0 DEVICE=cpu."; \
			exit 1; \
		}; \
		echo "[OK] PyTorch CUDA is available."; \
	fi

visual-train: check-gpu
	python -m src.visual.models.train --data-yaml $(DATA_YAML) --model $(MODEL) --epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH) --device $(DEVICE) --workers $(WORKERS) --project $(VISUAL_PROJECT) --name $(RUN_NAME)

visual-pipeline: check-gpu
	python -m src.visual.pipelines.training_pipeline --dataset-root $(DATASET_ROOT) --date-folder $(DATE_FOLDER) --cameras $(CAMERAS) --split-ratios $(SPLIT_RATIOS) --seed $(SEED) --model $(MODEL) --epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH) --device $(DEVICE) --workers $(WORKERS) --run-name $(RUN_NAME) --experiment-name $(VISUAL_EXPERIMENT_NAME)

mbt-run:
	python main.py multimodal --run-name $(MBT_RUN_NAME) --overrides $(MBT_OVERRIDES)

mbt-multi:
	@if [ -n "$(MBT_EXPERIMENTS_FILE)" ]; then \
		if [ "$(MBT_STOP_ON_ERROR)" = "1" ]; then \
			python main.py multimodal-batch --experiments-file $(MBT_EXPERIMENTS_FILE) --base-overrides $(MBT_BASE_OVERRIDES) --stop-on-error; \
		else \
			python main.py multimodal-batch --experiments-file $(MBT_EXPERIMENTS_FILE) --base-overrides $(MBT_BASE_OVERRIDES); \
		fi; \
	else \
		if [ "$(MBT_STOP_ON_ERROR)" = "1" ]; then \
			python main.py multimodal-batch --experiments $(MBT_EXPERIMENTS) --base-overrides $(MBT_BASE_OVERRIDES) --stop-on-error; \
		else \
			python main.py multimodal-batch --experiments $(MBT_EXPERIMENTS) --base-overrides $(MBT_BASE_OVERRIDES); \
		fi; \
	fi

# Launch MLflow to see experiment results
ui:
	bash scripts/mlflow.sh

# Cleanup
clean:
	rm -rf .dvc/tmp/cache
	find . -type d -name "__pycache__" -exec rm -rf {} +