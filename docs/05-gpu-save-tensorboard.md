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

出现 “expected all tensors to be on the same device” 时，检查模型、输入和标签的 device。

## 保存模型

常用方式是保存 `state_dict`：

```python
torch.save(model.state_dict(), "model.pt")
```

加载时先创建同结构模型：

```python
model = TinyClassifier()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

继续训练时保存优化器状态：

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

安装依赖后在训练循环里写指标：

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

TensorBoard 适合观察 loss、accuracy、过拟合和学习率设置。
