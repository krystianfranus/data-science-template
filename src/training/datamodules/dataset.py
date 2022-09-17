import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.x = data["x"].to_numpy(dtype="float32")[:, None]
        self.x = torch.from_numpy(self.x)

        self.y = data["y"].to_numpy(dtype="float32")
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
