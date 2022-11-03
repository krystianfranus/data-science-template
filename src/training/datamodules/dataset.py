import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data["user"].to_numpy())
        self.items = torch.tensor(data["item"].to_numpy())
        self.targets = torch.tensor(data["target"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        target = self.targets[idx]
        return user, item, target


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
        self.users = torch.tensor(data["user"].to_numpy())
        self.items_neg = torch.tensor(data["item_neg"].to_numpy())
        self.items_pos = torch.tensor(data["item_pos"].to_numpy())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item_neg = self.items_neg[idx]
        item_pos = self.items_pos[idx]
        return user, item_neg, item_pos
