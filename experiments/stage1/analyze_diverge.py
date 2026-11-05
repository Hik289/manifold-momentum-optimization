import json
d = json.load(open("results_sciml.json"))
for task in ["Burgers-PINN", "Poisson-PINN"]:
    for bl in ["B3", "B5"]:
        b = d[task]["baselines"][bl]
        h = b["history"]
        best, final = b["best"], b["final"]
        opt = b["optimizer"]
        bs, br, bstep = bl, best["relative_L2"], best["step"]
        fr, fstep = final["relative_L2"], final["step"]
        print(f"{task} {bl} ({opt}): best relL2={br:.4e}@step{bstep}, "
              f"final relL2={fr:.3e}@step{fstep}")
        dv = None
        for r in h:
            if r["relative_L2"] > 1.0:
                dv = r
                break
        if dv:
            print(f"   diverge relL2>1 first@step{dv['step']} "
                  f"loss={dv['total_loss']:.2e}")
        else:
            print("   never diverged (relL2<=1 throughout)")
