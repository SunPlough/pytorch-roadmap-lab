from __future__ import annotations

import torch


def main() -> None:
    torch.manual_seed(7)
    x = torch.linspace(-2, 2, steps=80).unsqueeze(1)
    y = 3.0 * x - 0.5 + 0.15 * torch.randn_like(x)

    weight = torch.randn(1, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    lr = 0.08

    for step in range(1, 81):
        prediction = x @ weight + bias
        loss = ((prediction - y) ** 2).mean()

        loss.backward()

        with torch.no_grad():
            weight -= lr * weight.grad
            bias -= lr * bias.grad
            weight.grad.zero_()
            bias.grad.zero_()

        if step % 20 == 0:
            print(
                f"step={step:03d} loss={loss.item():.4f} "
                f"weight={weight.item():.3f} bias={bias.item():.3f}"
            )


if __name__ == "__main__":
    main()

