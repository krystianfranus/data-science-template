from itertools import islice

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data.iloc[:, 0].to_numpy())
        self.items = torch.tensor(data.iloc[:, 1].to_numpy())
        self.targets = torch.tensor(data.iloc[:, 2].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        target = self.targets[idx]
        return user, item, target


class CustomIterableDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def _mapper(self, line):
        user_id, item_id, target = line.strip("\n").split(",")
        user_id = torch.tensor(int(user_id))
        item_id = torch.tensor(int(item_id))
        target = torch.tensor(float(target))
        return user_id, item_id, target

    def __iter__(self):
        file_itr = open(self.filename, mode="r")
        mapped_itr = map(self._mapper, file_itr)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            mapped_itr = islice(mapped_itr, worker_id, None, num_workers)

        return mapped_itr


# class PredictDataset(Dataset):
#     def __init__(self, x: torch.tensor):
#         self.x = x
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx]


class BPRDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data.iloc[:, 0].to_numpy())
        self.items_neg = torch.tensor(data.iloc[:, 1].to_numpy())
        self.items_pos = torch.tensor(data.iloc[:, 2].to_numpy())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item_neg = self.items_neg[idx]
        item_pos = self.items_pos[idx]
        return user, item_neg, item_pos
