import math
import torch
from torch._C import device
import torch.nn as nn
import numpy as np


class TCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel=3, dropout=0.5):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv2 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv3 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.bn = nn.BatchNorm2d(out_features)
        self.dropout = dropout

    def forward(self, inputs):
        """
        param inputs: (batch_size, timestamp, num_node, in_features)
        return: (batch_size, timestamp - 2, num_node, out_features)
        """
        inputs = inputs.permute(0, 3, 2, 1)  # (btnf->bfnt)
        out = torch.relu(self.conv1(inputs)) * \
            torch.sigmoid(self.conv2(inputs))
        #out = torch.relu(out + self.conv3(inputs))
        out = self.bn(out)
        out = out.permute(0, 3, 2, 1)
        out = torch.dropout(out, p=self.dropout, train=self.training)
        return out


class TCN(nn.Module):
    def __init__(self, n_history, in_features, mid_features) -> None:
        super(TCN, self).__init__()
        # odd time seriers: number of layer is n_hitory // 2
        # even time seriers: number of layer is n_history//2-1 + a single conv layer.
        # -> Aggregate information from time seriers to one unit
        assert(n_history >= 3)
        self.is_even = False if n_history % 2 != 0 else True

        self.n_layers = n_history // \
            2 if n_history % 2 != 0 else (n_history // 2 - 1)

        self.tcn_layers = nn.ModuleList([TCNLayer(in_features, mid_features)])
        for i in range(self.n_layers - 1):
            self.tcn_layers.append(TCNLayer(mid_features, mid_features))

        if self.is_even:
            self.tcn_layers.append(
                TCNLayer(mid_features, mid_features, kernel=2))

        self.upsample = None if in_features == mid_features else nn.Conv2d(
            in_features, mid_features, kernel_size=1)

    def forward(self, inputs):
        out = self.tcn_layers[0](inputs)
        if self.upsample:
            ResConn = self.upsample(inputs.permute(0, 3, 2, 1))
            ResConn = ResConn.permute(0, 3, 2, 1)
        else:
            ResConn = inputs

        out = out + ResConn[:, 2:, ...]

        tmp = [out[:]]
        for i in range(1, self.n_layers):
            out = self.tcn_layers[i](
                out) + out[:, 2:, ...] + ResConn[:, 2 * (i+1):, ...]

        if self.is_even:
            out = self.tcn_layers[-1](out) + out[:, -1,
                                                 :, :].unsqueeze(1) + ResConn[:, -1:, ...]

        return out


class Encoder(nn.Module):
    def __init__(self, n_history, n_predict, in_features, mid_features) -> None:
        super(Encoder, self).__init__()
        assert(n_history >= 3)
        self.n_predict = n_predict

        self.tcn = TCN(n_history=n_history,
                       in_features=in_features, mid_features=mid_features)

        self.fully = nn.Linear(mid_features, n_predict * mid_features)

        self.reset_parameter()

    def reset_parameter(self):
        for param in self.fully.parameters():
            param.data.normal_()

    def forward(self, inputs):
        out = self.tcn(inputs)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.relu(self.fully(out))
        out = out.reshape(out.shape[0], out.shape[1], self.n_predict, -1)
        out = out.permute(0, 2, 1, 3)
        return out


class GCNCell(nn.Module):
    def __init__(self, in_features, out_features, cheb_k=3, dropout=0.5) -> None:
        super(GCNCell, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))

        self.cheb_k = cheb_k
        self.dropout = dropout

        self.reset_parameter()

    def reset_parameter(self):
        self.weight.data.normal_()
        self.bais.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamp, num_node, num_features)
        adj: (num_node, num_node)
        """
        supports = [torch.eye(adj.shape[0]).to(adj.device), adj]
        for k in range(2, self.cheb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])

        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports,
                           inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bais)
        return result.permute(0, 2, 1, 3)


class Archer(nn.Module):
    def __init__(self, num_node, num_embed, n_history, n_predict, in_features, mid_features, out_features, adj) -> None:
        super(Archer, self).__init__()
        self.pre_linear = nn.Linear(in_features, mid_features)

        self.adj = nn.Parameter(torch.from_numpy(adj), requires_grad=False)
        self.embed = nn.Parameter(torch.FloatTensor(num_node, num_embed))

        self.encoder = Encoder(n_history, n_predict,
                               mid_features, mid_features)

        self.gcn = GCNCell(mid_features, mid_features)

        self.gcn_with_embed = GCNCell(mid_features, mid_features)
        self.leakyRelu = nn.ReLU()

        self.downsample = nn.Conv2d(
            2 * mid_features, mid_features, kernel_size=1)

        self.fully = nn.Linear(mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

        self.reset_parameter()

    def reset_parameter(self):
        self.embed.data.normal_()

    def forward(self, inputs):
        out = torch.relu(self.pre_linear(inputs))

        out = self.encoder(out)

        ResConn = out

        dynamic_adj = torch.softmax(self.leakyRelu(torch.matmul(
            self.embed, self.embed.T)), dim=1)
        embed_out = self.gcn_with_embed(out, dynamic_adj)

        out = self.gcn(out, self.adj)

        out = torch.cat([embed_out, out], dim=-1)

        out = out.permute(0, 3, 2, 1)
        out = self.downsample(out)
        out = out.permute(0, 3, 2, 1)

        out = out + ResConn

        out = torch.relu(self.fully(out))
        out = self.predict(out)
        return out


if __name__ == '__main__':
    adj = np.random.randn(307, 307).astype(np.float32)
    data = torch.randn(64, 12, 307, 1)
    net = Archer(12, 12, 1, 32, 1, adj)
    total = 0
    for param in net.parameters():
        total += param.numel()
    print(total)
