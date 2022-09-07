import torch
import numpy as np
from options import args
from models.model_v6 import Archer
from models.STGCN import STGCN
from datareader.data_util import tDataUtil


import matplotlib.pylab as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataPath = {
    'NPEMS04': "./dataset/NEWPEMS/PEMS04/PEMS04.npz",
    'NPEMS07': "./dataset/NEWPEMS/PEMS07/PEMS07.npz"
}


def normlize_data(data):
    data = data.transpose((1, 2, 0))

    mean = np.mean(data, axis=(0, 2))
    data -= mean.reshape((1, -1, 1))
    std = np.std(data, axis=(0, 2))
    data /= std.reshape((1, -1, 1))
    mean = mean[0]
    std = std[0]
    data = data.transpose((2, 0, 1))

    return data, mean, std


def unnormalize_data(data, mean, std):
    return data * std + mean


if __name__ == "__main__":

    args.dataset = "NPEMS07"

    data_util = tDataUtil(args)
    print(data_util.num_node)

    archer = Archer(data_util.num_node, 9, args.n_history,
                    args.n_predict, 1, 32, 1, data_util.adj)

    archer.load_state_dict(torch.load(
        f'./best_model_params/net_params_{args.dataset}_best.pkl'))

    stgcn = STGCN(data_util.num_node, num_features=1, num_timesteps_input=args.n_history,
                  num_timesteps_output=args.n_predict, adj=data_util.adj)

    stgcn.load_state_dict(torch.load(
        f'./best_model_params/net_params_{args.dataset}_STGCN_best.pkl'))

    inputs = []
    targets = []
    results = []
    stgcnResult = []
    data = np.load(dataPath[args.dataset])['data'][..., :1]
    data = data.astype(np.float32)
    data, mean, std = normlize_data(data)

    for i in range(0, 16968, 12):
        inputs.append(data[i:i+12, ...])
        targets.append(data[i+12:i+24, ...])

    for src in inputs:
        results.append(archer(torch.from_numpy(
            src).unsqueeze(0)).detach().numpy())
        stgcnResult.append(stgcn(torch.from_numpy(
            src).unsqueeze(0)).detach().numpy())

    results = np.stack(results, axis=0)[:, 0, ...]
    stgcnResult = np.stack(stgcnResult, axis=0)[:, 0, ...]
    targets = np.stack(targets, axis=0)

    results = unnormalize_data(results, mean, std)
    stgcnResult = unnormalize_data(stgcnResult, mean, std)
    targets = unnormalize_data(targets, mean, std)

    results = results.reshape(-1, data_util.num_node, 1)
    stgcnResult = stgcnResult.reshape(-1, data_util.num_node, 1)
    targets = targets.reshape(-1, data_util.num_node, 1)

    for j in range(0, data_util.num_node, 2):
        plt.xlabel("Time")
        plt.ylabel("Flow")
        # plt.plot(results[288*5:6*288, j], label="predict")
        plt.plot(targets[288*5:6*288, j], label="traffic flow")
        # plt.plot(stgcnResult[288*5:6*288, j], label="stgcn_predict")
        plt.legend(loc="best")
        plt.savefig(f"imgs/{args.dataset}/contrast_{j}.png")
        plt.clf()
