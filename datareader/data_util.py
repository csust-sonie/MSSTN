from torch.utils.data.dataloader import DataLoader
from datareader.reader import data_factory
from datareader.normal_dataset import DatasetNormal
from datareader.with_preweek_dataset import DatasetWithPreWeek


class tDataUtil:
    def __init__(self, args) -> None:
        data, self.mean, self.std, self.adj = data_factory(args.dataset)
        self.num_node = self.adj.shape[0]

        total_len = len(data)
        train_len = int(total_len * args.train_rate)
        self.train_loader = self._generator_loader(
            data[:train_len], args, shuffle=True)

        test_start, test_end = int(train_len), int(
            train_len + total_len * args.test_rate)
        self.test_loader = self._generator_loader(
            data[test_start:test_end], args, shuffle=False)

        valid_start, valid_end = int(test_end), int(total_len)
        self.valid_loader = self._generator_loader(
            data[valid_start:valid_end], args, shuffle=True)

        del data

    def unnormal(self, data):
        return data * self.std + self.mean

    @staticmethod
    def _generator_loader(data, args, shuffle):
        if args.sample == 'normal':
            dataset = DatasetNormal(data, args.n_history, args.n_predict)
        elif args.sample == 'preweek':
            dataset = DatasetWithPreWeek(data, args.n_history, args.n_predict)
        else:
            raise ValueError("Choice the sample function in [normal, preweek]")

        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=True)
        return data_loader


if __name__ == '__main__':
    class args(object):
        pass

    args.dataset = 'NPEMS04'
    args.train_rate = 0.7
    args.test_rate = 0.15
    args.valid_rate = 0.15
    args.n_history = 12
    args.n_predict = 12
    args.batch_size = 64
    args.sample = 'normal'

    dataUtil = tDataUtil(args)
    print(len(dataUtil.train_loader))
    print(len(dataUtil.test_loader))
    print(len(dataUtil.valid_loader))

    for src, tgt in dataUtil.train_loader:
        print(src[0].shape, src[1].shape)
        print(tgt.shape)

        # print("normal data: {}", src)
        # print("unnormal data: {}", dataUtil.unnormal(src))
        break
