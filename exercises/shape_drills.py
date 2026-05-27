from __future__ import annotations

import torch


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    """Convert [batch, channels, height, width] into [batch, features]."""
    # TODO: replace the next line.
    raise NotImplementedError


def select_last_token(hidden_states: torch.Tensor) -> torch.Tensor:
    """Convert [batch, sequence, hidden] into [batch, hidden]."""
    # TODO: replace the next line.
    raise NotImplementedError


def pairwise_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a matrix where output[i, j] is the dot product of a[i] and b[j]."""
    # TODO: replace the next line.
    raise NotImplementedError


def _self_check() -> None:
    images = torch.randn(4, 3, 8, 8)
    hidden = torch.randn(2, 5, 16)
    a = torch.randn(3, 10)
    b = torch.randn(6, 10)

    assert flatten_images(images).shape == (4, 192)
    assert select_last_token(hidden).shape == (2, 16)
    assert pairwise_dot(a, b).shape == (3, 6)
    print("all shape drills passed")


if __name__ == "__main__":
    _self_check()

