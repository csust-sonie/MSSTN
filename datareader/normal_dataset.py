import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class DatasetNormal(Dataset):
    def __init__(self, data: np.ndarray, n_history: int, n_predict: int) -> None:
        super(DatasetNormal, self).__init__()

        self.data_x = []
        self.data_y = []
        self.total_inputs = len(data) - n_history - n_predict + 1
        for start in range(self.total_inputs):
            self.data_x.append(data[start:start+n_history])
            self.data_y.append(data[start+n_history:start+n_history+n_predict])
        self.data_x = torch.from_numpy(np.array(self.data_x, dtype=np.float32))
        self.data_y = torch.from_numpy(np.array(self.data_y, dtype=np.float32))

        del data

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.total_inputs
