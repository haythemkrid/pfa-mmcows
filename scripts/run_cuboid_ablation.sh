#!/usr/bin/env bash
# scripts/run_cuboid_ablation.sh
#
# Usage:
#   bash scripts/run_cuboid_ablation.sh [--sanity-only] [--frames all|test|0-39]
#
# What this does
# ──────────────
# 1. Activates the shared conda environment (pfa-mmcows-env)
# 2. Runs a 2-frame sanity check and saves diagnostic figures
# 3. If --sanity-only is NOT set, launches the full ablation in a detached
#    tmux session called "cuboid_ablation"
#
# Resuming after a crash
# ──────────────────────
# The ablation loop checkpoints every condition to
#   logs/cuboid_ablation/checkpoints/<condition_name>.csv
# Re-running this script skips already-completed conditions automatically.
#
# Monitoring progress
# ───────────────────
#   tmux attach -t cuboid_ablation
#   tail -f logs/cuboid_ablation/cuboid_ablation.log
#   cat logs/cuboid_ablation/ablation_results.csv

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV="pfa-mmcows-env"
SESSION="cuboid_ablation"
DATASET_ROOT="${MULTIVIEWX_ROOT:-$HOME/pfa-mmcows/store/data/raw/MultiviewX}"
OUTPUT_DIR="logs/cuboid_ablation"
MLFLOW_URI="sqlite:///logs/mlflow/mlflow.db"
FRAMES="${FRAMES:-all}"
SEED=42

# Parse arguments
SANITY_ONLY=0
for arg in "$@"; do
    case $arg in
        --sanity-only) SANITY_ONLY=1 ;;
        --frames=*)    FRAMES="${arg#*=}" ;;
        --frames)      shift; FRAMES="$1" ;;
    esac
done

# ── Activate conda ────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        echo "[WARN] Could not activate conda env '$CONDA_ENV' — using current Python."
    }
fi

PYTHON=$(command -v python3 || command -v python)
echo "[INFO] Python: $PYTHON ($(${PYTHON} --version 2>&1))"
echo "[INFO] Dataset root: $DATASET_ROOT"
echo "[INFO] Output dir:   $OUTPUT_DIR"
echo "[INFO] Frames:       $FRAMES"

# ── Dependency check ──────────────────────────────────────────────────────────
echo "[INFO] Checking dependencies …"
$PYTHON - <<'EOF'
from importlib import util
import sys
missing = []
for pkg in ["cv2", "numpy", "pandas", "scipy", "matplotlib", "mlflow"]:
    if util.find_spec(pkg) is None:
        missing.append(pkg)
if missing:
    print(f"[ERROR] Missing packages: {missing}")
    print("        Run: pip install opencv-python pandas scipy matplotlib mlflow")
    sys.exit(1)
print("[INFO] All dependencies OK.")
EOF

# ── Sanity check (always runs in foreground first) ─────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  SANITY CHECK (2 frames, baseline)"
echo "════════════════════════════════════════════════════════"
mkdir -p "$OUTPUT_DIR"

$PYTHON -m src.cuboid_ablation.run_ablation \
    --dataset-root "$DATASET_ROOT" \
    --output-dir   "$OUTPUT_DIR" \
    --frames       "all" \
    --seed         "$SEED" \
    --mlflow-uri   "$MLFLOW_URI" \
    --sanity-only \
    2>&1 | tee "$OUTPUT_DIR/sanity_check.log"

echo ""
echo "[INFO] Sanity figures saved to $OUTPUT_DIR/sanity/"
echo "[INFO] Review them at:  ls $OUTPUT_DIR/sanity/"

if [ "$SANITY_ONLY" -eq 1 ]; then
    echo "[INFO] --sanity-only set — exiting."
    exit 0
fi

# ── Ask for confirmation before full run ──────────────────────────────────────
echo ""
read -rp "Sanity check done. Launch full ablation? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "[INFO] Aborted by user."
    exit 0
fi

# ── Launch full ablation in tmux ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  LAUNCHING FULL ABLATION in tmux session: $SESSION"
echo "════════════════════════════════════════════════════════"

# Kill stale session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

LOGFILE="$OUTPUT_DIR/cuboid_ablation.log"

tmux new-session -d -s "$SESSION" \
    "source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null; \
     conda activate $CONDA_ENV 2>/dev/null || true; \
     $PYTHON -m src.cuboid_ablation.run_ablation \
         --dataset-root '$DATASET_ROOT' \
         --output-dir   '$OUTPUT_DIR' \
         --frames       '$FRAMES' \
         --seed         $SEED \
         --mlflow-uri   '$MLFLOW_URI' \
         2>&1 | tee '$LOGFILE'; \
     echo 'ABLATION COMPLETE — press any key'; read"

echo ""
echo "[INFO] Ablation running in tmux session '$SESSION'."
echo ""
echo "  Monitor live:    tmux attach -t $SESSION"
echo "  Tail log:        tail -f $LOGFILE"
echo "  Detach from tmux: Ctrl+B then D"
echo "  Results table:   $OUTPUT_DIR/ablation_results.csv"
echo "  Figures:         $OUTPUT_DIR/figures/"
