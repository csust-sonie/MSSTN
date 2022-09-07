from ast import arg
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.model_v9 import Archer
from datareader.data_util import tDataUtil
from options import args

if __name__ == '__main__':
    data_util = tDataUtil(args)
    net = Archer(data_util.num_node, 9, args.n_history, args.n_predict, 1, 32, 1, data_util.adj)

    leakReLU = torch.nn.LeakyReLU()
    dadj = torch.softmax(leakReLU(torch.matmul(
        net.embed, net.embed.T
    )), dim=1)

    dadj = dadj.detach().numpy()
    plt.matshow(dadj, cmap=plt.cm.Reds)
    plt.title("dynamic adjacent")
    plt.savefig("./dynamicAdj.png")
    plt.clf()

    dadjPart = dadj[100:200, 100:200]
    plt.matshow(dadjPart, cmap=plt.cm.Reds)
    plt.title("dynamic adjacent part")
    plt.savefig("./dynamicAdjPart.png")
    plt.clf()

    sadjPart = data_util.adj[:100, :100]
    plt.matshow(sadjPart, cmap=plt.cm.Reds)
    plt.title("static adjacent part")
    plt.savefig("./staticAdjPart.png")

    