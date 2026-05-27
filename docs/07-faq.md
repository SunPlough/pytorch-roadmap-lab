# 07. FAQ

## NumPy 和 PyTorch 的学习顺序

如果已经熟悉数组、矩阵乘法、广播和索引，可以直接学 PyTorch。相关概念不熟时，先补 NumPy 数组操作会更顺。

入门阶段重点是 shape：`[batch, features]`、`[batch, classes]`、`[batch, channels, height, width]`。

## labels 为什么常用 long

分类任务里，`CrossEntropyLoss` 需要类别编号，而不是 one-hot 向量。类别编号通常是 `torch.long`：

```python
labels = labels.long()
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

如果 labels 是浮点数，检查任务类型是否为分类。

## 训练集准确率上涨，验证集不涨

常见原因是过拟合或数据切分问题：

- 训练集和验证集是否来自同一分布。
- 训练集是否太小，模型是否太大。
- 验证时是否调用了 `model.eval()` 和 `torch.no_grad()`。

## CPU 环境

入门阶段的大部分概念可以在 CPU 上完成。GPU 主要用于加速大模型和大数据训练。

本仓库的玩具任务默认可在 CPU 上运行。

## 官方文档

建议先跑通一个完整例子，再读官方文档。此时 `Dataset`、`Module`、`Optimizer` 等概念会更容易对应到代码。
