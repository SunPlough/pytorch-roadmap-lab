# PyTorch Roadmap Lab

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

PyTorch 入门笔记和配套代码。内容围绕一个二维分类任务展开，覆盖 tensor、autograd、Dataset/DataLoader、`nn.Module`、训练循环、模型保存和常见排错。

![decision boundary preview](assets/decision-boundary-preview.png)

## Contents

| No. | Note | Topic |
| --- | --- | --- |
| 01 | [Tensor 基础](docs/01-tensors.md) | shape、dtype、device、reshape |
| 02 | [Autograd](docs/02-autograd.md) | 计算图、梯度、`zero_grad()` |
| 03 | [Dataset 与 DataLoader](docs/03-dataset-dataloader.md) | dataset、batch、shuffle、split |
| 04 | [模型与训练循环](docs/04-model-training.md) | `nn.Module`、loss、optimizer |
| 05 | [GPU、保存与可视化](docs/05-gpu-save-tensorboard.md) | device、checkpoint、TensorBoard |
| 06 | [排错清单](docs/06-debug-checklist.md) | shape、device、loss、gradient |
| 07 | [FAQ](docs/07-faq.md) | 环境、学习顺序、常见问题 |

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python examples/01_tensor_playground.py
python examples/02_autograd_probe.py
python examples/03_train_toy_classifier.py
```

`03_train_toy_classifier.py` 会训练一个小 MLP，并把 checkpoint 写到 `artifacts/tiny_classifier.pt`。

训练完成后可以画分类边界：

```bash
python scripts/plot_decision_boundary.py
```

没有安装 PyTorch 时，可以只绘制 README 里的预览图：

```bash
python scripts/render_readme_preview.py
```

## Layout

```text
.
├── docs/                         # notes
├── examples/                     # runnable examples
├── exercises/                    # TODO exercises
├── src/pytorch_roadmap_lab/      # dataset, model, training helpers
└── tests/                        # pytest tests
```

## Exercises

- `exercises/shape_drills.py`: tensor shape 练习。
- `exercises/custom_dataset_todo.py`: 自定义 Dataset 练习。

## References

- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
- [torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
