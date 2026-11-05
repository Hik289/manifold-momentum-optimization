"""Small deterministic CPU demonstration of the optimizer interface."""
import torch
from torch import nn
from torch.nn import functional as F

from bures.optimizer import MatrixOptimizer


def main():
    torch.manual_seed(42)
    x = torch.randn(256, 16)
    y = ((x[:, 0] + 0.7 * x[:, 1] - 0.5 * x[:, 2]) > 0).long()
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 2))
    optimizer = MatrixOptimizer(model.parameters(), lr=3e-3, covariance_rule="bures")
    initial = F.cross_entropy(model(x), y).item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    final = F.cross_entropy(model(x), y).item()
    print(f"BuresMomentum CPU demo: initial loss={initial:.4f}, final loss={final:.4f}")
    assert torch.isfinite(torch.tensor(final))


if __name__ == "__main__":
    main()
