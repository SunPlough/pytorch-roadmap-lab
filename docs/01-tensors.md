# 01. Tensor 基础

Tensor 可以理解为带有类型和设备信息的多维数组。调试时优先检查三项信息：

- `shape`: 数据的结构，比如 `[batch, channels, height, width]`。
- `dtype`: 数据类型，比如 `float32`、`int64`。
- `device`: 数据在哪个设备上，比如 `cpu` 或 `cuda:0`。

```python
import torch

x = torch.randn(8, 3, 32, 32)
print(x.shape)   # torch.Size([8, 3, 32, 32])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

## shape

模型层只接收 tensor，不关心变量名。常见分类批次的形状：

```text
features: [batch_size, num_features]
labels:   [batch_size]
logits:   [batch_size, num_classes]
```

`CrossEntropyLoss` 报错时先打印：

```python
print(logits.shape, labels.shape, logits.dtype, labels.dtype)
```

`CrossEntropyLoss` 通常要求 logits 为浮点数，labels 为类别 id，类型为 `torch.long`。

## view、reshape、permute

- `reshape` 改变维度布局，适合把图像摊平成向量。
- `permute` 改变维度顺序，常见于通道位置转换。
- `unsqueeze` 增加一个维度，常见于把单个样本变成 batch。

```python
images = torch.randn(16, 3, 28, 28)
flat = images.reshape(16, -1)
single = flat[0].unsqueeze(0)

print(flat.shape)    # [16, 2352]
print(single.shape)  # [1, 2352]
```

## shape 注释

模型代码里可以保留关键 shape 注释：

```python
# x: [batch, 2]
# logits: [batch, 2]
logits = model(x)
```

复杂模型里这类注释能减少维度错误。
