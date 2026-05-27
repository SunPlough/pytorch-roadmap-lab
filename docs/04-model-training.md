# 04. 模型与训练循环

PyTorch 模型通常继承 `nn.Module`。`__init__` 声明层，`forward` 描述前向计算。

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

四个步骤：

- forward: 计算预测。
- loss: 计算标量损失。
- backward: 计算梯度。
- step: 更新参数。

## train 和 eval

`model.train()` 和 `model.eval()` 会影响 Dropout、BatchNorm 等层的行为。

```python
model.train()
# training loop

model.eval()
with torch.no_grad():
    # validation loop
```

## 记录指标

基础指标通常记录 loss 和 accuracy。分类任务还可以按需加入 precision、recall、F1、AUC。
