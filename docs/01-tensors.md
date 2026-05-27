# 01. Tensor 基础

PyTorch 里的 tensor 可以先理解成“带设备信息的多维数组”。初学时不要急着背 API，先把每个 tensor 的三件事看清楚：

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

## 为什么 shape 是第一优先级

模型不关心变量名叫 `image` 还是 `input`，它只关心最后收到的 tensor 形状是否符合预期。一个常见训练批次通常长这样：

```text
features: [batch_size, num_features]
labels:   [batch_size]
logits:   [batch_size, num_classes]
```

如果 `CrossEntropyLoss` 报错，第一件事就是打印：

```python
print(logits.shape, labels.shape, logits.dtype, labels.dtype)
```

`CrossEntropyLoss` 通常期望 logits 是浮点数，labels 是类别 id，也就是 `torch.long`。

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

## 一个实用习惯

写模型前先写 shape 注释：

```python
# x: [batch, 2]
# logits: [batch, 2]
logits = model(x)
```

这比“我感觉这里应该对”可靠得多。

