from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


def make_two_rings(
    n_samples: int = 1024,
    noise: float = 0.08,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small binary classification dataset without external downloads."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n_samples,), generator=generator)
    angles = 2 * math.pi * torch.rand(n_samples, generator=generator)
    base_radius = torch.where(labels == 0, 0.75, 1.45).float()
    radius = base_radius + noise * torch.randn(n_samples, generator=generator)

    x = torch.stack(
        [
            radius * torch.cos(angles),
            radius * torch.sin(angles),
        ],
        dim=1,
    )
    return x.float(), labels.long()


class ToyClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A Dataset wrapper around tensors returned by make_two_rings."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_samples, num_features]")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [num_samples]")
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of samples")
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

