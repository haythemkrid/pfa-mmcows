# pfa-mmcows

Projet ML structure pour MMCows (sensor + visual + fusion).

## Migration visual depuis mmcows-visual

Le travail visual a ete migre dans ce repo, sans modifier le repo source.

Scripts principaux migres:
- src/visual/data/a1_build_index.py
- src/visual/data/a2_yolo_binary.py
- src/visual/data/remap_labels_single_class.py
- src/visual/models/train.py
- src/visual/pipelines/training_pipeline.py

## Prerequis

1. Installer les dependances:

```bash
pip install -r requirements.txt
```

2. Verifier que les donnees visual sont sous:

```text
store/data/raw/visual_data/
	images/0725/cam_1 ... cam_4
	labels/0725/cam_1 ... cam_4
```

## Execution rapide

1. Construire index dataset:

```bash
make visual-index
```

2. Generer artefacts YOLO (train/val/test + yaml):

```bash
make visual-yolo
```

3. Entrainer YOLOv8:

```bash
make visual-train
```

Ou tout en une commande:

```bash
make visual-pipeline
```

## DVC

Le pipeline DVC inclut 3 stages visual:
- visual_build_index
- visual_prepare_yolo
- visual_train

Lancer:

```bash
dvc repro
```
