import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data["user"].to_numpy())
        self.items = torch.tensor(data["item"].to_numpy())
        self.ratings = torch.tensor(data["rating"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        return user, item, rating


# class PredictDataset(Dataset):
#     def __init__(self, x: torch.tensor):
#         self.x = x
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx]
