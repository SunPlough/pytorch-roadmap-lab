# 04. 模型与训练循环

PyTorch 模型通常继承 `nn.Module`。你在 `__init__` 里声明层，在 `forward` 里描述数据怎么流过这些层。

```python
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
```

## 训练循环的四步

```python
for features, labels in train_loader:
    logits = model(features)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这四步顺序很重要：

- forward: 算预测。
- loss: 把预测和答案变成一个标量。
- backward: 根据 loss 算梯度。
- step: 优化器根据梯度改参数。

## train 和 eval

`model.train()` 和 `model.eval()` 不只是语义标记。Dropout、BatchNorm 这类层在训练和评估时行为不同。

```python
model.train()
# training loop

model.eval()
with torch.no_grad():
    # validation loop
```

## 记录指标

最小可用指标是 loss 和 accuracy。更复杂的任务可以加入 precision、recall、F1、AUC，但不要一开始就把指标系统做复杂。先确认训练真的在学习。

