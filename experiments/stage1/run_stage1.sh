#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p stage1

if ! "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "Torch is not available in $PYTHON"
  exit 1
fi

if ! "$PYTHON" -c "import torch_geometric" >/dev/null 2>&1; then
  echo "Install the graph extra before running Stage 1: python -m pip install -e '.[graph]'"
  exit 1
fi

GPU_C10=${GPU_C10:-0}
GPU_C100=${GPU_C100:-1}
GPU_SHARED=${GPU_SHARED:-2}

# Vision
CUDA_VISIBLE_DEVICES="$GPU_C10" "$PYTHON" stage1/train_vision.py --task CIFAR-10 --epochs 10 --gpu 0 &
PID_C10=$!
CUDA_VISIBLE_DEVICES="$GPU_C100" "$PYTHON" stage1/train_vision.py --task CIFAR-100 --epochs 10 --gpu 0 &
PID_C100=$!

# Graph
CUDA_VISIBLE_DEVICES="$GPU_SHARED" "$PYTHON" stage1/train_graph.py --task Cora --epochs 200 --gpu 0 &
PID_CORA=$!
CUDA_VISIBLE_DEVICES="$GPU_SHARED" "$PYTHON" stage1/train_graph.py --task Citeseer --epochs 200 --gpu 0 &
PID_CITESEER=$!
CUDA_VISIBLE_DEVICES="$GPU_SHARED" "$PYTHON" stage1/train_graph.py --task PubMed --epochs 200 --gpu 0 &
PID_PUBMED=$!

# SciML
CUDA_VISIBLE_DEVICES="$GPU_SHARED" "$PYTHON" stage1/train_sciml.py --task Burgers-PINN --steps 5000 --gpu 0 &
PID_BURGERS=$!
CUDA_VISIBLE_DEVICES="$GPU_SHARED" "$PYTHON" stage1/train_sciml.py --task Poisson-PINN --steps 5000 --gpu 0 &
PID_POISSON=$!

wait "$PID_C10" "$PID_C100" "$PID_CORA" "$PID_CITESEER" "$PID_PUBMED" "$PID_BURGERS" "$PID_POISSON"
echo "Stage 1 complete"
