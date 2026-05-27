from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FeatureLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wrap two tensors so DataLoader can batch them."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        # TODO: validate dimensions and save tensors on self.
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: return number of samples.
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: return one feature row and one label.
        raise NotImplementedError


def _self_check() -> None:
    features = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10,))
    dataset = FeatureLabelDataset(features, labels)
    x0, y0 = dataset[0]

    assert len(dataset) == 10
    assert x0.shape == (4,)
    assert y0.ndim == 0
    print("custom dataset exercise passed")


if __name__ == "__main__":
    _self_check()

