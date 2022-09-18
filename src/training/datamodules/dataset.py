import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.x = data[["x"]].to_numpy(dtype=np.float32)
        self.x = torch.from_numpy(self.x)
        self.y = data["y"].to_numpy(dtype=np.float32)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PredictDataset(Dataset):
    def __init__(self, x: torch.tensor):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
