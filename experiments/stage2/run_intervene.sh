#!/usr/bin/env bash
# Stage2 决定性机制实验: 尺度-only INTERVENE 三臂对照 (抗 gateway 打断)
set -u
cd "$(dirname "$0")"
S2=$(pwd)
echo "[intervene] start $(date '+%F %T %Z')"
python -c "import torch;print('[intervene] torch',torch.__version__,'cuda',torch.cuda.is_available())"

GPU=${GPU:-0}

echo "===== sanity: scale_only == bures 在共基情形 (§4.3) ====="
python - <<'PY'
import torch, sys
sys.path.insert(0, "..")
from bures.core import (symmetrize, bures_barycenter_2pt,
                        scale_only_average, spd_sqrt)
torch.manual_seed(42)
# 共基: A,B 用同一正交基 -> scale_only 应 == 完整 Bures (§4.3)
d=16; Q,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64))
a=torch.rand(d,dtype=torch.float64)+0.2; b=torch.rand(d,dtype=torch.float64)+0.2
A=symmetrize(Q@torch.diag(a)@Q.T); B=symmetrize(Q@torch.diag(b)@Q.T)
for al in (0.01,0.3,0.7):
    bu=bures_barycenter_2pt(A,B,al); so=scale_only_average(A,B,al)
    r=(bu-so).norm().item()/(bu.norm().item())
    print(f"  commuting alpha={al}: bures vs scale_only relerr={r:.2e}",
          "OK" if r<1e-6 else "MISMATCH")
# 非共基: 二者应有差异 (scale_only 不旋转)
A2=symmetrize(torch.randn(d,d,dtype=torch.float64)@torch.randn(d,d,dtype=torch.float64).T)+0.1*torch.eye(d)
B2=symmetrize(torch.randn(d,d,dtype=torch.float64)@torch.randn(d,d,dtype=torch.float64).T)+0.1*torch.eye(d)
bu=bures_barycenter_2pt(A2,B2,0.3); so=scale_only_average(A2,B2,0.3)
r=(bu-so).norm().item()/(bu.norm().item())
print(f"  non-commuting alpha=0.3: relerr={r:.2e}",
      "OK(有差异,符合预期)" if r>1e-4 else "WARN(无差异?)")
PY

echo "===== INTERVENE 三臂 (Cora, 400ep) ====="
python run_scale_only_intervene.py --task Cora --epochs 400 --seed 42 \
  --gpu $GPU --diag-interval 25 \
  --output "$S2/results_scale_only_intervene.json" 2>&1
echo "[intervene] done rc=$?  $(date '+%F %T %Z')"
