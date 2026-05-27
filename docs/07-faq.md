# 07. FAQ

## 我应该先学 NumPy 还是直接学 PyTorch

如果你已经知道数组、矩阵乘法、广播和索引，直接学 PyTorch 没问题。如果这些概念还很模糊，建议先补 NumPy 的数组操作，再回到 PyTorch。

PyTorch 初学阶段最值得练的是 shape 感。你不需要马上记住所有函数，但要能看懂 `[batch, features]`、`[batch, classes]`、`[batch, channels, height, width]` 这些结构。

## 为什么 labels 经常要用 long

分类任务里，`CrossEntropyLoss` 需要类别编号，而不是 one-hot 向量。类别编号通常是 `torch.long`：

```python
labels = labels.long()
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

如果 labels 是浮点数，先检查你是不是把回归任务和分类任务混在了一起。

## 为什么训练集准确率能涨，验证集不涨

这通常是过拟合或数据切分问题。先做三个检查：

- 训练集和验证集是否来自同一分布。
- 训练集是否太小，模型是否太大。
- 验证时是否调用了 `model.eval()` 和 `torch.no_grad()`。

## CPU 能不能学 PyTorch

能。初学阶段的大部分概念都可以在 CPU 上学清楚。GPU 主要让大模型和大数据训练更快，但不会替你理解 shape、loss 和梯度。

本仓库的玩具任务默认可以在 CPU 上快速跑完。

## 我该什么时候读官方文档

建议先跑通一个完整例子，再读官方文档。跑通以后，你读到 `Dataset`、`Module`、`Optimizer` 时会有具体经验可以挂靠，理解速度会快很多。

