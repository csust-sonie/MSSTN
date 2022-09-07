
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import math


def subsequent_mask(size):
    "Mask out subsequent positions."
    atten_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    subsequent_mask = (subsequent_mask == 0)
    return torch.from_numpy(subsequent_mask)


class GraphCN(nn.Module):
    def __init__(self, in_features, out_features, cheb_k=3, dropout=0.5) -> None:
        super(GraphCN, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.chb_k = cheb_k
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamp, num_node, num_features)
        adj: (num_node, num_node)
        """
        supports = [torch.eye(adj.shape[0]).to(adj.device), adj]
        for k in range(2, self.chb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])

        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports,
                           inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bias)

        return result.permute(0, 2, 1, 3)


class SelfAttention(nn.Module):
    def __init__(self, num_node, seq_len, in_features, n_head, n_dim_per_head, act=torch.relu) -> None:
        super(SelfAttention, self).__init__()

        D = n_head * n_dim_per_head
        self.n_dim_per_head = n_dim_per_head

        self.LinearTrans = nn.Conv2d(
            in_features, n_dim_per_head, kernel_size=(1, 1))

        self.GraphQ = GraphCN(n_dim_per_head, D)
        self.GraphK = GraphCN(n_dim_per_head, D)
        self.GraphV = GraphCN(n_dim_per_head, D)

        self.LowSampleLinear = nn.Conv2d(D, n_dim_per_head, kernel_size=(1, 1))

        self.LayerNormal = nn.LayerNorm([n_dim_per_head, seq_len, num_node])

        self.act = act

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamp, num_node, in_features)
        """
        batch_size = inputs.shape[0]

        X = inputs.permute(0, 3, 1, 2)
        # (batch_size, in_feature, timestamp, num_node)

        X = self.act(self.LinearTrans(X))
        # (batch_size, n_dim_per_head, timestamp, num_node)
        ResCon = X

        X = X.permute(0, 2, 3, 1)
        # (batch_size, timestamp, num_node, n_dim_per_haed)

        Q = self.GraphQ(X, adj)
        K = self.GraphK(X, adj)
        V = self.GraphV(X, adj)

        Q = Q.permute(0, 3, 1, 2)
        K = K.permute(0, 3, 1, 2)
        V = V.permute(0, 3, 1, 2)
        # (batch_size, n_dim_per_head * head, timestamp, num_node)

        Q = torch.cat(torch.split(Q, self.n_dim_per_head, dim=1), dim=0)
        K = torch.cat(torch.split(K, self.n_dim_per_head, dim=1), dim=0)
        V = torch.cat(torch.split(V, self.n_dim_per_head, dim=1), dim=0)
        # (batch_size * n_head, n_dim_per_head, timestamp, num_node)

        Q = Q.permute(0, 3, 2, 1)
        # (batch_size * n_head, num_node, timestamp, n_dim_per_head)
        K = K.permute(0, 3, 1, 2)
        # (batch_size * n_head, num_node, n_dim_per_head, timestamp)

        attention = torch.matmul(Q, K) / (self.n_dim_per_head ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        mask = subsequent_mask(attention.shape[-1])
        mask = mask.to(attention.device)

        attention = attention.masked_fill(mask == 0, 1e-9)

        V = V.permute(0, 3, 2, 1)
        # (batch_size * n_head, num_node, timestamp, n_dim_per_head)

        result = torch.matmul(attention, V)

        result = self.act(result)

        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)
        # (batch_size, num_node, timestamp, n_dim_per_head * n_head)

        result = result.permute(0, 3, 2, 1)
        # (batch_size, n_dim_per_head * n_head, timestamp, num_node)
        result = self.act(self.LowSampleLinear(result))
        # (batch_size, n_dim_per_head, timestamp, num_node)

        result = self.LayerNormal(ResCon + result)

        result = result.permute(0, 2, 3, 1)
        # (batch_size, timestamp, num_node, n_dim_per_head)

        return result


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_features, mid_feautre, seq_len, num_node, dropout=0.5) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.linOne = nn.Linear(in_features, mid_feautre)
        self.linSec = nn.Linear(mid_feautre, in_features)
        self.dropout = nn.Dropout(dropout)

        self.LayerNorm = nn.LayerNorm([in_features, seq_len, num_node])

    def forward(self, inputs):
        """
        inputs: (batch_size, timestamp, num_node, in_features)
        """
        out = self.dropout(torch.relu(self.linOne(inputs)))
        out = inputs + self.linSec(out)

        out = out.permute(0, 3, 1, 2)
        out = self.LayerNorm(out)
        out = out.permute(0, 2, 3, 1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, in_features, seq_len, dropout=0.5) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, in_features)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_features, 2)
                             * -(math.log(10000.0) / in_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, inputs):
        """
        inputs: (batch_size, timestamp, num_node, in_features)
        """
        out = inputs.permute(0, 2, 1, 3)
        out = out + Variable(self.pe[:, :inputs.shape[1]], requires_grad=False)
        return self.dropout(out.permute(0, 2, 1, 3))


class EncoderLayer(nn.Module):
    def __init__(self, num_node, in_seq_len, n_head, in_feautre, n_features_per_head, dropout=0.5) -> None:
        super(EncoderLayer, self).__init__()

        self.position_embedding = PositionalEncoding(
            in_feautre, in_seq_len, dropout)
        self.mutli_attn = SelfAttention(
            num_node, in_seq_len, in_feautre, n_head, n_features_per_head)
        self.feedForward = PositionwiseFeedForward(
            n_features_per_head, n_features_per_head*2, in_seq_len, num_node, dropout)

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamps, num_node, in_features);
        """
        out = self.position_embedding(inputs)
        out = self.mutli_attn(out, adj)
        out = self.feedForward(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_features, mid_features, out_features, num_node, in_seq_len, n_head=8, n_layer=2, dropout=0.5) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(EncoderLayer(num_node, in_seq_len,
                           n_head, in_features, mid_features, dropout))

        for i in range(1, n_layer):
            self.layers.append(EncoderLayer(
                num_node, in_seq_len, n_head, mid_features, mid_features, dropout))

        self.LinearOut = nn.Linear(mid_features, out_features)

    def forward(self, inputs, adj):
        out = inputs
        for layer in self.layers:
            out = layer(out, adj)

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 1)
    adj = torch.randn(307, 307)

    encoder = Encoder(1, 32, 1, 307, 12)

    encoder(data, adj)
