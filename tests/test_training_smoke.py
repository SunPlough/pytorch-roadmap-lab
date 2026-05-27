import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, random_split

from pytorch_roadmap_lab import TinyClassifier, ToyClassificationDataset, fit, make_two_rings


def test_tiny_training_loop_runs() -> None:
    features, labels = make_two_rings(n_samples=96, noise=0.1, seed=7)
    dataset = ToyClassificationDataset(features, labels)
    train_dataset, val_dataset = random_split(
        dataset,
        [72, 24],
        generator=torch.Generator().manual_seed(7),
    )

    model = TinyClassifier(hidden_size=8)
    history = fit(
        model,
        DataLoader(train_dataset, batch_size=24, shuffle=True),
        DataLoader(val_dataset, batch_size=24),
        epochs=2,
        lr=1e-2,
        device=torch.device("cpu"),
    )

    assert len(history.train_loss) == 2
    assert len(history.val_acc) == 2
    assert all(0.0 <= value <= 1.0 for value in history.val_acc)

