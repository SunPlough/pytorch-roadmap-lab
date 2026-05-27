from __future__ import annotations

import torch


def describe(name: str, tensor: torch.Tensor) -> None:
    print(f"{name:>12} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")


def main() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.ones_like(x)
    z = x + 2 * y

    describe("x", x)
    describe("y", y)
    describe("z", z)
    describe("z[:, :2]", z[:, :2])
    describe("z.T", z.T)

    batch = torch.randn(8, 3, 32, 32)
    describe("image batch", batch)
    print("batch mean:", batch.mean().item())
    print("per-channel mean:", batch.mean(dim=(0, 2, 3)))


if __name__ == "__main__":
    main()

