<h1 align="center">BuresMomentum</h1>
<p align="center"><strong>Bures–Wasserstein covariance averaging for matrix preconditioning</strong></p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0D8479.svg" alt="MIT License"></a>
</p>

<p align="center">
  <img src="assets/fig_buresmomentum_method.png" width="96%" alt="BuresMomentum method: gradient and first moment, left/right covariance statistics, Bures transport and averaging, spectral clipping, two-sided preconditioning, and parameter update; dashed Euclidean EMA baseline">
</p>

## Method

BuresMomentum maintains separate positive-definite covariance states for the rows and columns of each matrix parameter. Each gradient supplies a new statistic for both states. The update moves each state toward its statistic along a Bures–Wasserstein geodesic, then bounds its spectrum:

$$
C_t=\mathrm{ClipSPD}\!\left(\mathrm{Bar}_{\mathrm{BW}}(C_{t-1},S_t;\,1-\beta_2)\right).
$$

Here $C$ denotes either covariance state, and $S_t$ its current gradient statistic. The inverse square roots of the updated states precondition the bias-corrected first moment on both sides. A bounded Frobenius-norm match sets the direction's scale before the parameter step.

Setting `covariance_rule="euclidean"` replaces only the Bures average with an arithmetic moving average. The gradient, first moment, spectral clipping, preconditioner, norm match, and parameter update use the same path in both variants.

## Installation

Python 3.10+ and PyTorch 2.1+ are required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For tabular, vision, language, plotting, and data-preparation experiments:

```bash
python -m pip install -e ".[experiments]"
```

Graph experiments additionally require PyTorch Geometric, installed for the chosen PyTorch/CUDA environment:

```bash
python -m pip install -e ".[graph]"
```

## Quick start

```python
import torch
from bures.optimizer import MatrixOptimizer

model = torch.nn.Linear(32, 8)
optimizer = MatrixOptimizer(
    model.parameters(), lr=3e-4,
    covariance_rule="bures",  # or "euclidean" for the paired baseline
)

optimizer.zero_grad(set_to_none=True)
loss = model(torch.randn(16, 32)).square().mean()
loss.backward()
optimizer.step()
```

A deterministic CPU example and the geometric invariant checks run without downloading datasets:

```bash
python examples/quickstart.py
python experiments/tests/test_numerical.py --out /tmp/bures-numerical.json
```

The numerical suite checks SPD symmetry/positivity, transport identities, geodesics, the covariance update, block behavior, cache consistency, and mixed precision.

## Development

```bash
python -m pip install -e ".[test]"
python experiments/tests/test_numerical.py --out /tmp/bures-numerical.json
python examples/quickstart.py
```

The GitHub Actions workflow runs these CPU checks on every push and pull request. The equation-rich [technical method schematic](assets/fig_buresmomentum_update_detailed.png) can be regenerated with:

```bash
python -m pip install matplotlib
python assets/draw_method_update.py
```

## License

Code is released under the [MIT License](LICENSE). Dataset licenses remain with their respective sources.
