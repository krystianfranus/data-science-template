from itertools import islice

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler


class SimpleDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data["user"].to_numpy())
        self.items = torch.tensor(data["item"].to_numpy())
        self.targets = torch.tensor(data["target"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.targets[idx]


class UserGroupedDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.unique_users = list(self.data["user"].unique())
        self.user_groups = {
            user: data[data["user"] == user].index.tolist()
            for user in self.unique_users
        }  # indices per user

    def __len__(self):
        return len(self.unique_users)

    def __getitem__(self, idx):
        user = self.unique_users[idx]  # Get user at index
        user_indices = self.user_groups[user]  # Get all rows for this user
        user_data = self.data.iloc[user_indices]  # Fetch data for this user

        users = torch.tensor(user_data["user"].to_numpy())
        items = torch.tensor(user_data["item"].to_numpy())
        targets = torch.tensor(user_data["target"].to_numpy(), dtype=torch.float32)
        return users, items, targets  # Return user-wise batch


class UserBatchSampler(Sampler):
    def __init__(self, dataset):
        self.unique_users = dataset.unique_users

    def __iter__(self):
        for i in range(len(self.unique_users)):
            yield [i]  # Yield dataset indices for each user

    def __len__(self):
        return len(self.unique_users)


def collate_fn(batch):
    users, items, targets = zip(*batch)
    return users[0], items[0], targets[0]


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
        file_itr = open(self.filename)
        mapped_itr = map(self._mapper, file_itr)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            mapped_itr = islice(mapped_itr, worker_id, None, num_workers)

        return mapped_itr


class BPRDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data.iloc[:, 0].to_numpy())
        self.items_neg = torch.tensor(data.iloc[:, 1].to_numpy())
        self.items_pos = torch.tensor(data.iloc[:, 2].to_numpy())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items_neg[idx], self.items_pos[idx]


class InferDataset(Dataset):
    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items

    def __len__(self):
        return self.n_users * self.n_items

    def __getitem__(self, idx: int):
        user = idx // self.n_users
        item = idx % self.n_items
        return user, item
