"""Report peak SPD condition numbers and divergence for SciML runs."""
import json
import math
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results_sciml.json"
d = json.load(open(RES))


def fmax(a):
    v = [x for x in a if x is not None
         and not (isinstance(x, float) and math.isnan(x))]
    return max(v) if v else None


def analyze(task, bl):
    if task not in d or bl not in d[task]["baselines"]:
        print(f"  {task} {bl}: MISSING")
        return
    b = d[task]["baselines"][bl]
    diag = b["diagnostics"]
    kL = [x.get("kappa_L_max") or x.get("kappa_L") for x in diag]
    kR = [x.get("kappa_R_max") or x.get("kappa_R") for x in diag]
    lmL = [x.get("lambda_min_L") for x in diag]
    lmR = [x.get("lambda_min_R") for x in diag]
    pkL, pkR = fmax(kL), fmax(kR)
    opt = b["optimizer"]
    print(f"=== {task} {bl} ({opt}) ===")
    print(f"  final relL2={b['final']['relative_L2']:.4e}  "
          f"best={b['best']['relative_L2']:.4e}")
    print(f"  kappa_L peak={pkL:.3e}" if pkL else "  kappa_L peak=None")
    print(f"  kappa_R peak={pkR:.3e}" if pkR else "  kappa_R peak=None")
    lminL = [x for x in lmL if x is not None]
    lminR = [x for x in lmR if x is not None]
    if lminL:
        print(f"  lambda_min_L range=[{min(lminL):.2e},{max(lminL):.2e}]")
    if lminR:
        print(f"  lambda_min_R range=[{min(lminR):.2e},{max(lminR):.2e}]")
    # Find the first recorded divergence.
    diverged = False
    for x in diag:
        tl = x.get("train_loss")
        if tl is not None and isinstance(tl, float) and \
                (math.isnan(tl) or abs(tl) > 1e6):
            print(f"  >> DIVERGE@step={x['step']}: loss={tl:.3e} "
                  f"kL={x.get('kappa_L_max')} kR={x.get('kappa_R_max')} "
                  f"scale={x.get('scale_mean')} "
                  f"gnorm={x.get('raw_grad_norm')} "
                  f"precond={x.get('precond_update_norm')}")
            diverged = True
            break
    if not diverged:
        print("  >> no divergence (train_loss finite throughout)")
    # Show the last three diagnostic records.
    for x in diag[-3:]:
        print(f"    step={x['step']} loss={x.get('train_loss')} "
              f"kL={x.get('kappa_L_max')} kR={x.get('kappa_R_max')} "
              f"scale={x.get('scale_mean')} gnorm={x.get('raw_grad_norm')}")


for task in ["Burgers-PINN", "Poisson-PINN"]:
    for bl in ["B3", "B5"]:
        analyze(task, bl)
    print()
