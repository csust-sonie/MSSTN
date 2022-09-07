from cProfile import label
from matplotlib.pyplot import plot
import torch
import numpy as np
from options import args
from datareader.data_util import tDataUtil

import matplotlib.pylab as plt


dataPath = {
    'NPEMS04' : "./dataset/NEWPESM/PEMS04/PEMS04.npz",
    'NPEMS07': "./dataset/NEWPEMS/PEMS07/PEMS07.npz"
}

def normalize_data(data: np.ndarray):
    data = data.transpose((1,2,0))

    mean = np.mean(data, axis=(0, 2))
    data -= mean.reshape((1, -1, 1))
    std = np.std(data, axis=(0, 2))
    data /= std.reshape((1, -1, 1))
    mean = mean[0]
    std = std[0]
    data = data.transpose((2, 0, 1))

    return data, mean, std

if __name__ == '__main__':
    
    args.dataset = "NPEMS07"

    data_util = tDataUtil(args)
    print(data_util.num_node)

    data = np.load(dataPath[args.dataset])['data'][..., :1]
    data = data.astype(np.float32)
    
    for i in range(0, data_util.num_node, 2):
        plt.xlabel("Time")
        plt.ylabel("Flow")
        plt.plot(data[5*288:6*288, i], label="traffic flow")
        plt.legend(loc="best")
        plt.savefig(f"imgs/true/ground_true_{i}.png")
        plt.clf()
    