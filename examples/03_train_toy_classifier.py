from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from pytorch_roadmap_lab import (
    TinyClassifier,
    ToyClassificationDataset,
    count_trainable_parameters,
    fit,
    make_two_rings,
    select_device,
    set_seed,
)


def main() -> None:
    set_seed(42)
    features, labels = make_two_rings(n_samples=1200, noise=0.12, seed=42)
    dataset = ToyClassificationDataset(features, labels)
    train_dataset, val_dataset = random_split(
        dataset,
        [960, 240],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = TinyClassifier(hidden_size=32)
    device = select_device()
    print(f"device={device} trainable_parameters={count_trainable_parameters(model)}")

    history = fit(model, train_loader, val_loader, epochs=25, lr=3e-3, device=device)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    checkpoint_path = artifacts_dir / "tiny_classifier.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history.__dict__,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

