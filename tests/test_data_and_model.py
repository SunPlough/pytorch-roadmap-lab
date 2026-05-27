import pytest

torch = pytest.importorskip("torch")

from pytorch_roadmap_lab import TinyClassifier, ToyClassificationDataset, make_two_rings


def test_make_two_rings_shapes() -> None:
    features, labels = make_two_rings(n_samples=20, seed=123)

    assert features.shape == (20, 2)
    assert labels.shape == (20,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.long


def test_dataset_returns_one_sample() -> None:
    features, labels = make_two_rings(n_samples=8, seed=123)
    dataset = ToyClassificationDataset(features, labels)
    x, y = dataset[0]

    assert len(dataset) == 8
    assert x.shape == (2,)
    assert y.ndim == 0


def test_model_forward_shape() -> None:
    model = TinyClassifier(in_features=2, hidden_size=8, num_classes=2)
    logits = model(torch.randn(5, 2))

    assert logits.shape == (5, 2)

