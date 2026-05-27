# Exercises

这些练习故意很小。真正的目标不是写很多代码，而是练出 PyTorch 初学阶段最重要的三个反射：

- 先看 shape，再看变量名。
- 先确认张量在哪个 device，再看报错。
- 先检查 `requires_grad` 和 `zero_grad()`，再怀疑优化器。

建议顺序：

1. `shape_drills.py`
2. `custom_dataset_todo.py`
3. 回到 `examples/03_train_toy_classifier.py`，自己加 TensorBoard 或 loss 曲线。

