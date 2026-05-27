# 03. Dataset 与 DataLoader

`Dataset` 负责“按索引拿一个样本”，`DataLoader` 负责“把很多样本组装成 batch”。

一个最小 Dataset：

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

再交给 DataLoader：

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True)

for features, labels in loader:
    print(features.shape, labels.shape)
```

## shuffle 该怎么用

- 训练集通常 `shuffle=True`，让模型不要按固定顺序看数据。
- 验证集通常 `shuffle=False`，便于复现实验和排查问题。

## batch size 的取舍

batch size 不是越大越好。太小会让训练曲线抖动明显，太大可能显存不够，也可能让模型更新不够灵活。

初学时可以从 32、64、128 这几个数试起。

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

固定 seed 不是迷信，它能让你下次复查问题时看到同样的数据切分。

