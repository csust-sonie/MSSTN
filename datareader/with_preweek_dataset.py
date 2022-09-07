import torch
import numpy as np
from torch.utils.data.dataset import Dataset

TimestamplePerWeek = 12 * 24 * 7


class DatasetWithPreWeek(Dataset):
    def __init__(self, data: np.ndarray, n_history: int, n_predict: int) -> None:
        super(DatasetWithPreWeek, self).__init__()
        self.data_x = []
        self.data_y = []
        self.total_inputs = len(data) - n_history - \
            n_predict + 1 - TimestamplePerWeek

        rate_of_preweek = 0
        for start in range(TimestamplePerWeek, self.total_inputs + TimestamplePerWeek):
            if start != 0 and start % TimestamplePerWeek == 0:
                rate_of_preweek += 1
            preweek_data = data[rate_of_preweek *
                                TimestamplePerWeek: rate_of_preweek * TimestamplePerWeek + n_history + n_predict]
            self.data_x.append(
                np.concatenate([preweek_data, data[start: start+n_history]], axis=0))
            self.data_y.append(
                data[start + n_history: start + n_history + n_predict])

        self.data_x = torch.from_numpy(np.array(self.data_x, dtype=np.float32))
        self.data_y = torch.from_numpy(np.array(self.data_y, dtype=np.float32))

        del data

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.total_inputs


def Split_last_week_filter(data, n_history, n_predict):
    last_week = data[:, :n_history + n_predict, ...]
    last_week_now = last_week[:, :n_history, ...]
    last_week_predict = last_week[:, n_history:n_history+n_predict, ...]
    now_history = data[:, n_history+n_predict:, ...]
    return last_week_now, last_week_predict, now_history
