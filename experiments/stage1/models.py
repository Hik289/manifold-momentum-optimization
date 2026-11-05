import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover - graph script reports the real import error.
    GCNConv = None


def ResNet18Small(num_classes):
    from torchvision import models as tv_models

    model = tv_models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SmallGCN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, dropout=0.5):
        super().__init__()
        if GCNConv is None:
            raise ImportError(
                "torch_geometric is required for SmallGCN. Install it with "
                "`pip install torch_geometric` in the experiment environment."
            )
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class PINNNet(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=4, out_dim=1):
        super().__init__()
        if layers < 2:
            raise ValueError("PINNNet requires at least two layers")
        modules = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MLPTabular(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
