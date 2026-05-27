# 02. Autograd

Autograd 是 PyTorch 的自动求导系统。前向计算得到 loss 后，调用 `loss.backward()` 即可沿计算图反向传播梯度。

最小流程：

```python
prediction = model(features)
loss = criterion(prediction, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## requires_grad

模型参数默认需要梯度：

```python
for name, parameter in model.named_parameters():
    print(name, parameter.requires_grad)
```

输入数据通常不需要梯度。分类训练里主要更新模型参数，而不是原始输入。

## 为什么要 zero_grad

PyTorch 的梯度默认累加。常规训练里需要每个 step 清空上一轮梯度，否则后续更新会叠加历史梯度。

常用顺序：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 什么时候用 no_grad

评估阶段不需要记录计算图：

```python
with torch.no_grad():
    logits = model(features)
```

`torch.no_grad()` 可以节省显存，并避免评估计算进入训练图。

## 常见信号

- `loss.backward()` 报错：检查 loss 是否是标量，检查参与计算的 tensor 是否断开了图。
- 参数没有变化：检查 optimizer 是否拿到了 `model.parameters()`，检查是否调用了 `optimizer.step()`。
- 梯度全是 `None`：检查参数是否真的参与了 loss 的计算。
