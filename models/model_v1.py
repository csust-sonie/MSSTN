import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from models.gcn_module import GCNBlock


class GRUCell(nn.Module):
    def __init__(self, num_seq, in_features, out_features) -> None:
        super(GRUCell, self).__init__()
        self.Wxr, self.Whr, self.br = self.get_param(
            num_seq, in_features, out_features)
        self.Wxr, self.Whr, self.br = nn.Parameter(
            self.Wxr), nn.Parameter(self.Whr), nn.Parameter(self.br)

        self.Wxz, self.Whz, self.bz = self.get_param(
            num_seq, in_features, out_features)
        self.Wxz, self.Whz, self.bz = nn.Parameter(
            self.Wxz), nn.Parameter(self.Whz), nn.Parameter(self.bz)

        self.Wxh, self.Whh, self.bh = self.get_param(
            num_seq, in_features, out_features)
        self.Wxh, self.Whh, self.bh = nn.Parameter(
            self.Wxh), nn.Parameter(self.Whh), nn.Parameter(self.bh)

    @staticmethod
    def get_param(num_seq, in_features, out_features):
        x = torch.from_numpy(np.random.normal(
            0, 1, size=(num_seq, in_features, out_features)).astype(np.float32))
        h = torch.from_numpy(np.random.normal(
            0, 1, size=(num_seq, out_features, out_features)).astype(np.float32))
        b = torch.zeros((num_seq, out_features), dtype=torch.float32)
        return x, h, b

    def forward(self, inputs, hidden_state):
        """
        param inputs: (batch_size, num_node, in_features)
        param hidden_state: (batch_size, num_node, out_features)
        return: (batch_size, num_node, out_features)
        """
        torch.einsum('bnf,nfo->bno', inputs, self.Wxr)

        torch.einsum('bnf,nfo->bno', hidden_state, self.Whr)
        Rt = F.relu(torch.einsum('bnf,nfo->bno', inputs, self.Wxr) +
                    torch.einsum('bnf,nfo->bno', hidden_state, self.Whr) + self.br)
        Zt = F.relu(torch.einsum('bnf,nfo->bno', inputs, self.Wxz) +
                    torch.einsum('bnf,nfo->bno', hidden_state, self.Whz) + self.bz)

        H_hat_t = torch.tanh(torch.einsum('bnf,nfo->bno', inputs, self.Wxh) +
                             torch.einsum('bnf,nfo->bno', Rt * hidden_state, self.Whh) + self.bh)
        H_t = Zt * hidden_state + (1 - Zt) * H_hat_t

        return torch.softmax(H_t, dim=-1)


class GRU(nn.Module):
    def __init__(self, n_layer, predict_len, num_node, in_features, mid_feautres, out_features) -> None:
        super(GRU, self).__init__()
        self.out_features = out_features
        self.n_layers = n_layer
        self.predict_len = predict_len
        self.gru = nn.ModuleList(
            [GRUCell(num_seq=num_node, in_features=in_features, out_features=mid_feautres)])
        for i in range(1, n_layer):
            self.gru.append(
                GRUCell(num_seq=num_node, in_features=mid_feautres, out_features=mid_feautres))
        self.linear = nn.Parameter(
            torch.from_numpy(np.random.normal(0, 1, size=(mid_feautres, predict_len * out_features)).astype(np.float32)))

    def forward(self, inputs):
        hidden_state = torch.zeros(
            inputs.shape[0], inputs.shape[2], self.out_features, dtype=inputs.dtype).to(inputs.device)
        seq_len = inputs.shape[1]
        hidden_state_seq = [hidden_state]
        for i in range(seq_len):
            state = hidden_state_seq[-1]
            for layer in range(self.n_layers):
                state = self.gru[layer](inputs[:, i, :, :], state)
            hidden_state_seq.append(state)

        # (bno, o(l*out_feature)->bn(l*out))
        out = torch.matmul(hidden_state_seq[-1], self.linear)
        return out.reshape(out.shape[0], out.shape[1], self.predict_len, -1).permute(0, 2, 1, 3)


class Archer(nn.Module):
    def __init__(self, num_node, seq_len, in_features, out_features, adj) -> None:
        super(Archer, self).__init__()
        self.dynamic_adj = nn.Parameter(
            torch.from_numpy(adj), requires_grad=True)  # 使用normlized邻接矩阵初始化的动态邻接矩阵

        WEEK_EMBEDDING_SIZE = 11
        self.week_embedding = nn.Parameter(
            torch.FloatTensor(num_node, WEEK_EMBEDDING_SIZE))

        self.gcn = GCNBlock(num_node, in_features, out_features)

        self.gru = GRU(2, num_node=num_node, predict_len=seq_len, in_features=out_features,
                       mid_feautres=out_features, out_features=in_features)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param, 0, 1)

    @ staticmethod
    def softmax_and_relu(tensor):
        return torch.softmax(torch.relu(tensor), dim=1)

    def forward(self, inputs):
        """
        inputs:(batch_size, timestamp, num_node, features)
        """
        inputs = inputs.permute(0, 2, 1, 3)

        emb = self.softmax_and_relu(
            torch.mm(self.week_embedding, self.week_embedding.T))
        d_adj_mixup = self.softmax_and_relu(torch.mm(self.dynamic_adj, emb))

        result = self.gcn(inputs, d_adj_mixup)
        result = result.permute(0, 2, 1, 3)
        out = self.gru(result)
        out = out + result

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 1)
    adj = np.random.randn(307, 307).astype(np.float32)
    net = GRU(2, 12, 307, 1, 32, 1)
    net2 = Archer(307, 12, 1, 1, adj)
    print(net2(data).shape)
    res = net(data)
    print(res.shape)
