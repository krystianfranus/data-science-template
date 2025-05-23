# from itertools import islice

import numpy as np
import pandas as pd
import torch

# from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, Sampler


class SimpleDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.list_ids = torch.tensor(data["list_id"].to_numpy())
        self.users = torch.tensor(data["user_idx"].to_numpy())
        self.items = torch.tensor(data["item_idx"].to_numpy())
        self.targets = torch.tensor(data["target"].to_numpy(), dtype=torch.float32)
        self.user_history = torch.from_numpy(np.array(data["user_history"].tolist()))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        return (
            self.list_ids[idx],
            self.users[idx],
            self.items[idx],
            self.targets[idx],
            self.user_history[idx],
        )


class ListGroupedDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.unique_list_ids = list(self.data["list_id"].unique())
        self.list_id_groups = {
            list_id: data[data["list_id"] == list_id].index.tolist()
            for list_id in self.unique_list_ids
        }  # indices per list_id

    def __len__(self):
        return len(self.unique_list_ids)

    def __getitem__(self, idx):
        list_id = self.unique_list_ids[idx]  # Get list_id at index
        list_id_indices = self.list_id_groups[list_id]  # Get all rows for this list_id
        list_id_data = self.data.iloc[list_id_indices]  # Fetch data for this list_id

        list_ids = torch.tensor(list_id_data["list_id"].to_numpy())
        users = torch.tensor(list_id_data["user_idx"].to_numpy())
        items = torch.tensor(list_id_data["item_idx"].to_numpy())
        targets = torch.tensor(list_id_data["target"].to_numpy(), dtype=torch.float32)
        user_histories = torch.from_numpy(
            np.array(list_id_data["user_history"].tolist())
        )
        return list_ids, users, items, targets, user_histories  # Return user-wise batch


class ListBatchSampler(Sampler):
    def __init__(self, dataset):
        self.unique_list_ids = dataset.unique_list_ids

    def __iter__(self):
        for i in range(len(self.unique_list_ids)):
            yield [i]  # Yield dataset indices for each list_id

    def __len__(self):
        return len(self.unique_list_ids)


def collate_fn(batch):
    list_ids, users, items, targets, user_histories = zip(*batch)
    return list_ids[0], users[0], items[0], targets[0], user_histories[0]


# class CustomIterableDataset(IterableDataset):
#     def __init__(self, filename):
#         self.filename = filename

#     def _mapper(self, line):
#         user_id, item_id, target = line.strip("\n").split(",")
#         user_id = torch.tensor(int(user_id))
#         item_id = torch.tensor(int(item_id))
#         target = torch.tensor(float(target))
#         return user_id, item_id, target

#     def __iter__(self):
#         file_itr = open(self.filename)
#         mapped_itr = map(self._mapper, file_itr)

#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is not None:
#             num_workers = worker_info.num_workers
#             worker_id = worker_info.id
#             mapped_itr = islice(mapped_itr, worker_id, None, num_workers)

#         return mapped_itr


class InferDataset(Dataset):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_history_based: bool,
        last_user_histories: pd.DataFrame,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.user_history_based = user_history_based
        if user_history_based:
            self.last_user_histories = last_user_histories.set_index("user_idx")
            self.last_user_histories = torch.from_numpy(
                np.array(last_user_histories["user_history"].tolist())
            )

    def __len__(self):
        return self.n_users * self.n_items

    def __getitem__(self, idx: int):
        user = idx // self.n_users
        if self.user_history_based:
            user = self.last_user_histories[user]
        item = idx % self.n_items
        return user, item
