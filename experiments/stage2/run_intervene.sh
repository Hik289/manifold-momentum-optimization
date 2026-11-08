#!/usr/bin/env bash
# Compare full BW, scale-only BW, and Euclidean averaging on Cora.
set -u
cd "$(dirname "$0")"
S2=$(pwd)
echo "[intervene] start $(date '+%F %T %Z')"
python -c "import torch;print('[intervene] torch',torch.__version__,'cuda',torch.cuda.is_available())"

GPU=${GPU:-0}

echo "===== Check: scale-only equals BW when matrices share eigenvectors ====="
python - <<'PY'
import torch, sys
sys.path.insert(0, "..")
from bures.core import (symmetrize, bures_barycenter_2pt,
                        scale_only_average, spd_sqrt)
torch.manual_seed(42)
# Shared eigenvectors should make scale-only and full BW agree.
d=16; Q,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64))
a=torch.rand(d,dtype=torch.float64)+0.2; b=torch.rand(d,dtype=torch.float64)+0.2
A=symmetrize(Q@torch.diag(a)@Q.T); B=symmetrize(Q@torch.diag(b)@Q.T)
for al in (0.01,0.3,0.7):
    bu=bures_barycenter_2pt(A,B,al); so=scale_only_average(A,B,al)
    r=(bu-so).norm().item()/(bu.norm().item())
    print(f"  commuting alpha={al}: bures vs scale_only relerr={r:.2e}",
          "OK" if r<1e-6 else "MISMATCH")
# Different eigenvectors should make the two averages differ.
A2=symmetrize(torch.randn(d,d,dtype=torch.float64)@torch.randn(d,d,dtype=torch.float64).T)+0.1*torch.eye(d)
B2=symmetrize(torch.randn(d,d,dtype=torch.float64)@torch.randn(d,d,dtype=torch.float64).T)+0.1*torch.eye(d)
bu=bures_barycenter_2pt(A2,B2,0.3); so=scale_only_average(A2,B2,0.3)
r=(bu-so).norm().item()/(bu.norm().item())
print(f"  non-commuting alpha=0.3: relerr={r:.2e}",
      "OK (different, as expected)" if r>1e-4 else "WARN (no difference)")
PY

echo "===== Compare three averaging rules on Cora for 400 epochs ====="
python run_scale_only_intervene.py --task Cora --epochs 400 --seed 42 \
  --gpu $GPU --diag-interval 25 \
  --output "$S2/results_scale_only_intervene.json" 2>&1
echo "[intervene] done rc=$?  $(date '+%F %T %Z')"
