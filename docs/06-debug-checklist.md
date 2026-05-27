# 06. PyTorch 排错清单

常见训练问题可以按下面顺序检查。

## 维度错误

打印四件事：

```python
print(features.shape, features.dtype, features.device)
print(labels.shape, labels.dtype, labels.device)
print(logits.shape, logits.dtype, logits.device)
print(loss)
```

常见情况：

- `Linear` 输入最后一维不对：检查 `in_features`。
- `CrossEntropyLoss` labels 形状不对：通常应是 `[batch]`。
- 图像通道顺序不对：确认是 `[batch, channels, height, width]`。

## device 错误

模型和数据必须在同一个 device：

```python
model.to(device)
features = features.to(device)
labels = labels.to(device)
```

features 和 labels 都需要移动到同一 device。

## loss 不下降

按顺序查：

1. 数据和标签是否对应。
2. learning rate 是否过大或过小。
3. 是否漏了 `optimizer.zero_grad()`。
4. 是否漏了 `loss.backward()` 或 `optimizer.step()`。
5. 模型最后一层输出维度是否等于类别数。

## 梯度异常

```python
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        print(name, "grad is None")
    else:
        print(name, parameter.grad.abs().mean().item())
```

如果大部分梯度是 `None`，检查参数是否参与 loss 计算。如果梯度特别大，可以尝试降低学习率或加梯度裁剪。

## 快速缩小问题

- 用 32 个样本做 overfit 测试。
- 把模型换成更小版本，确认训练循环可用。
- 固定 seed，复现同一个问题。
