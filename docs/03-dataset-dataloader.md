# 03. Dataset 与 DataLoader

`Dataset` 定义单个样本的读取方式，`DataLoader` 负责 batch、shuffle 和多进程加载。

最小 Dataset：

```python
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

交给 DataLoader：

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True)

for features, labels in loader:
    print(features.shape, labels.shape)
```

## shuffle 该怎么用

- 训练集通常 `shuffle=True`。
- 验证集通常 `shuffle=False`，便于复现。

## batch size 的取舍

batch size 会影响显存占用和梯度估计稳定性。

常用起点：32、64、128。

## 随机切分

```python
from torch.utils.data import random_split
import torch

train_set, val_set = random_split(
    dataset,
    [800, 200],
    generator=torch.Generator().manual_seed(42),
)
```

固定 seed 可以复现同样的数据切分。
