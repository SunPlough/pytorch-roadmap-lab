from pathlib import Path


def test_readme_points_to_learning_path() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "PyTorch Roadmap Lab" in readme
    assert "docs/01-tensors.md" in readme
    assert "examples/03_train_toy_classifier.py" in readme

