# 02. Autograd

Autograd 是 PyTorch 自动求导系统。你只要用 tensor 做前向计算，得到 loss，再调用 `loss.backward()`，PyTorch 就会沿着计算图把梯度算出来。

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

输入数据通常不需要梯度。训练普通分类模型时，我们关心的是如何更新参数，不是如何更新原始输入。

## 为什么要 zero_grad

PyTorch 的梯度默认累加。这很灵活，比如梯度累积训练会用到它；但普通训练里，如果不清空梯度，下一轮的梯度会叠在上一轮上，更新就会失真。

推荐顺序：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 什么时候用 no_grad

评估和手动更新参数时，不需要记录计算图：

```python
with torch.no_grad():
    logits = model(features)
```

`torch.no_grad()` 可以节省显存，也能避免把评估阶段的计算误接到训练图里。

## 常见信号

- `loss.backward()` 报错：检查 loss 是否是标量，检查参与计算的 tensor 是否断开了图。
- 参数没有变化：检查 optimizer 是否拿到了 `model.parameters()`，检查是否调用了 `optimizer.step()`。
- 梯度全是 `None`：检查参数是否真的参与了 loss 的计算。

