#!/usr/bin/env bash
# Stage2-P0 自包含后台脚本 (抗 gateway 打断)
# A1 消融(最高优先,先出) -> Cora 谱漂移 TRACE -> Cora/Citeseer 1000ep 长训练
set -u

cd "$(dirname "$0")"
S2=$(pwd)
echo "[stage2] start $(date '+%F %T %Z') dir=$S2"
python -c "import torch;print('[stage2] torch',torch.__version__,'cuda',torch.cuda.is_available())"

GPU=${GPU:-0}

echo "===================== TASK 1: A1 消融 (Cora, 决定 G4) ====================="
python run_A1_ablation.py --task Cora --epochs 200 --seed 42 --gpu $GPU \
  --output "$S2/results_A1_ablation.json" 2>&1
echo "[stage2] A1 done rc=$?"

echo "===================== TASK 2: Cora population 谱漂移 TRACE ================"
python trace_grad_cov.py --task Cora --epochs 200 --seed 42 --gpu $GPU \
  --checkpoints 1,10,25,50,100,150,200 \
  --output "$S2/results_trace_gradcov.json" 2>&1
echo "[stage2] TRACE done rc=$?"

echo "===================== TASK 3: 1000ep 长训练 (Cora + Citeseer) ============"
python "$S2/../stage1/train_graph.py" --task Cora --epochs 1000 --seed 42 \
  --gpu $GPU --diag-interval 50 --output "$S2/results_graph_1000ep.json" 2>&1
echo "[stage2] Cora 1000ep done rc=$?"
python "$S2/../stage1/train_graph.py" --task Citeseer --epochs 1000 --seed 42 \
  --gpu $GPU --diag-interval 50 --output "$S2/results_graph_1000ep.json" 2>&1
echo "[stage2] Citeseer 1000ep done rc=$?"

echo "[stage2] ALL DONE $(date '+%F %T %Z')"
