import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_features, out_features, cheb_k=3, dropout=0.5) -> None:
        super(GCN, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.cheb_k = cheb_k
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
        supports = [torch.eye(adj.shape[0]).to(inputs.device), adj]

        for _ in range(2, self.cheb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])

        supports = supports[1:]  # We not need the message of node self.
        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports,
                           inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bias)
        return result.permute(0, 2, 1, 3)


class StackGCN(nn.Module):
    def __init__(self, in_features, out_features, cheb_k=3, dropout=0.5) -> None:
        super(StackGCN, self).__init__()
        self.gcn = GCN(in_features, out_features,
                       cheb_k=cheb_k, dropout=dropout)

    def forward(self, inputs, adj):
        out = self.gcn(inputs, adj)
        out_time_len = out.shape[1]

        last_frame = out[:, -1, ...]
        outs = []
        for i in reversed(range(out_time_len-1)):
            outs.append(out[:, i, ...] + torch.relu(out[:, i+1, ...]))

        outs.append(last_frame)
        outs = torch.stack(outs, dim=1)

        return outs


class Encoder(nn.Module):
    def __init__(self, num_node, embed_dim, in_features, out_features) -> None:
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.gcn_static = StackGCN(out_features, out_features)
        self.gcn_dynamic = StackGCN(out_features, out_features)

        self.node_embedding = nn.Parameter(
            torch.FloatTensor(num_node, embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embedding.data.normal_()

    def forward(self, inputs, adj):
        out = torch.relu(self.linear(inputs))

        dynamic_adj = torch.softmax(torch.relu(
            torch.matmul(self.node_embedding, self.node_embedding.T)), dim=1)

        out_static = self.gcn_static(out, adj)
        out_dynamic = self.gcn_dynamic(out, dynamic_adj)

        out_one_stream_freame = []
        out_second_stream_freame = []

        len_of_seq = out_static.shape[1]
        for i in range(0, len_of_seq, 2):
            out_one_stream_freame.append(out_static[:, i, ...])
            out_one_stream_freame.append(out_dynamic[:, i+1, ...])
            out_second_stream_freame.append(out_dynamic[:, i, ...])
            out_second_stream_freame.append(out_static[:, i+1, ...])

        out_one = torch.stack(out_one_stream_freame, dim=1)
        out_two = torch.stack(out_second_stream_freame, dim=1)

        return out_one, out_two


class TCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3) -> None:
        super(TCNLayer, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_features, out_features,
                                              kernel_size=(kernel_size, 1)) for _ in range(3)])

    def forward(self, inputs):
        """
        param inputs: (batch_size, num_timestamp, num_node, in_features)
        return: (batch_size, num_timestamp - 2, num_node, out_features)
        """
        inputs = inputs.permute(0, 3, 1, 2)
        out = torch.tanh(self.convs[0](inputs)) + \
            torch.sigmoid(self.convs[1](inputs))
        out = self.convs[2](inputs) + out
        out = out.permute(0, 2, 3, 1)
        return out


class TCN(nn.Module):
    def __init__(self, in_seq_len, in_features, out_features) -> None:
        super(TCN, self).__init__()
        assert in_seq_len >= 3

        self.n_layers = in_seq_len // 2 if in_seq_len % 2 != 0 else in_seq_len // 2 - 1

        self.layers = nn.ModuleList(
            [TCNLayer(out_features, out_features) for _ in range(self.n_layers)])

        if in_seq_len % 2 == 0:
            self.layers.append(
                TCNLayer(out_features, out_features, kernel_size=2))

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)

        out += inputs[:, -1:, ...]
        return out


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, n_history) -> None:
        super(Decoder, self).__init__()
        self.tcn = TCN(n_history, in_features, out_features)

    def forward(self, inputs):
        """
        param inputs: (batch_size, n_history, num_node, in_features)
        """
        return self.tcn(inputs)


class Archer(nn.Module):
    def __init__(self, num_node, embed_dim, in_features, mid_features, out_features, n_history, n_predict, adj, device) -> None:
        super(Archer, self).__init__()
        self.adj = torch.from_numpy(adj).to(device)

        self.encoder = Encoder(num_node, embed_dim, in_features, mid_features)
        self.decoder_first = Decoder(
            mid_features, mid_features, n_history)
        self.decoder_second = Decoder(
            mid_features, mid_features, n_history)

        self.linear = nn.Linear(2 * mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

    def forward(self, inputs):
        out_first, out_second = self.encoder(inputs, self.adj)
        out_first = self.decoder_first(out_first)
        out_second = self.decoder_second(out_second)

        out = torch.concat([out_first, out_second], dim=-1)

        out = torch.relu(self.linear(out))
        out = inputs + out

        out = self.predict(out)

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 32)
    adj = torch.randn(307, 307)
    net = Archer(307, 9, 32, 32, 32, 12, 12, adj, "cpu")

    res = net(data)

    print(res.shape)
