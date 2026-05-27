# 05. GPU、保存与可视化

## 自动选择设备

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for features, labels in loader:
    features = features.to(device)
    labels = labels.to(device)
```

报错里出现 “expected all tensors to be on the same device” 时，基本就是模型、输入、标签不在同一个设备。

## 保存模型

推荐保存 `state_dict`，而不是直接保存整个模型对象：

```python
torch.save(model.state_dict(), "model.pt")
```

加载时需要先创建同结构模型：

```python
model = TinyClassifier()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

如果你还想继续训练，可以把优化器状态也保存：

```python
torch.save(
    {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    },
    "checkpoint.pt",
)
```

## TensorBoard

安装依赖后，可以在训练循环里写指标：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/tiny-classifier")
writer.add_scalar("loss/train", train_loss, epoch)
writer.add_scalar("acc/val", val_acc, epoch)
writer.close()
```

启动：

```bash
tensorboard --logdir runs
```

可视化不是为了好看，而是为了更快发现 loss 不降、过拟合、学习率过大这类问题。

