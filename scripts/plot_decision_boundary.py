from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from pytorch_roadmap_lab import TinyClassifier, ToyClassificationDataset, fit, make_two_rings, set_seed


def main() -> None:
    set_seed(42)
    features, labels = make_two_rings(n_samples=1200, noise=0.12, seed=42)
    dataset = ToyClassificationDataset(features, labels)
    train_dataset, val_dataset = random_split(
        dataset,
        [960, 240],
        generator=torch.Generator().manual_seed(42),
    )

    model = TinyClassifier(hidden_size=32)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    fit(model, train_loader, val_loader, epochs=25, lr=3e-3, device=torch.device("cpu"))

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-2.0, 2.0, 240),
        torch.linspace(-2.0, 2.0, 240),
        indexing="xy",
    )
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(grid), dim=1)[:, 1].reshape(grid_x.shape)

    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "decision-boundary.png"

    plt.figure(figsize=(8, 6), dpi=160)
    contour = plt.contourf(
        grid_x.numpy(),
        grid_y.numpy(),
        probabilities.numpy(),
        levels=24,
        cmap="RdYlBu_r",
        alpha=0.72,
    )
    plt.colorbar(contour, label="P(class=1)")
    plt.scatter(
        features[:, 0].numpy(),
        features[:, 1].numpy(),
        c=labels.numpy(),
        cmap="RdYlBu_r",
        s=12,
        edgecolors="white",
        linewidths=0.25,
    )
    plt.title("TinyClassifier decision boundary")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()

