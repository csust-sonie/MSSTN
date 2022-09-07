import torch
import torch.nn as nn
import numpy as np


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


class Encoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, in_features, out_features) -> None:
        super(Encoder, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_features, out_features, kernel_size=(1, 1), stride=(1, 1))

        self.tcn = TCN(in_seq_len, out_features, out_features)

        self.out_seq_len = out_seq_len
        self.linear = nn.Linear(out_features,
                                out_seq_len * out_features)

        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, inputs):
        """
        param inputs: (batch_size, timestamps, num_node, in_features)
        """
        out = inputs.permute(0, 3, 1, 2)
        out = torch.relu(self.conv1x1(out))
        out = out.permute(0, 2, 3, 1)

        out = self.tcn(out)

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.relu(self.linear(out))
        out = out.reshape(out.shape[0], out.shape[1],
                          self.out_seq_len, -1).permute(0, 2, 1, 3)

        return out


class Attention(nn.Module):
    def __init__(self, in_features, n_head, n_dim) -> None:
        super(Attention, self).__init__()
        D = n_head * n_dim

        self.n_head = n_head
        self.n_dim_per_head = n_dim

        self.conv_q = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.conv_v = nn.Conv2d(in_features, D, kernel_size=(1, 1))

        self.linear = nn.Linear(D, n_dim)

        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0, 3, 2, 1)

        # (batch_size, n_head * n_dim_per_head, num_node, timestamp)
        query = self.conv_q(inputs)
        key = self.conv_k(inputs)
        value = self.conv_v(inputs)

        # (n_head * batch_size, n_dim_per_head, num_node, timestamp)
        query = torch.concat(torch.split(
            query, self.n_dim_per_head, dim=1), dim=0)
        key = torch.concat(torch.split(key, self.n_dim_per_head, dim=1), dim=0)
        value = torch.concat(torch.split(
            value, self.n_dim_per_head, dim=1), dim=0)

        # (n_head * batch_size, timestamp, num_node, n_dim_per_head)
        query = query.permute(0, 3, 2, 1)

        # (n_head * batch_size,timestamp, n_dim_per_head,  num_node)
        key = key.permute(0, 3, 1, 2)

        # (n_head * batch_size,  timestamp, n_dim_per_head, num_node)
        value = value.permute(0, 3, 2, 1)

        attnetion = torch.matmul(query, key)
        attnetion /= self.n_dim_per_head ** 0.5
        attnetion = torch.softmax(attnetion, dim=-1)

        # (n_head * batch_size, timestamp, num_node, n_dim_per_head)
        result = torch.matmul(attnetion, value)
        # (n_head * batch_size, timestamp, num_node, n_dim_per_head)
        # result = result.permute(0, 2, 1, 3)
        # (batch_size, timestamp, num_node, n_head * n_dim_per_head)
        result = torch.concat(torch.split(result, batch_size, dim=0), dim=-1)

        result = torch.relu(self.linear(result))

        return result


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

        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports,
                           inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bias)
        return result.permute(0, 2, 1, 3)


class Archer(nn.Module):
    def __init__(self, num_node, in_seq_len, out_seq_len, in_features, mid_features, out_features, adj, device, embed_features=9, n_head=4, cheb_k=3, dropout=0.5) -> None:
        super(Archer, self).__init__()

        self.out_seq_len = out_seq_len

        self.encoder = Encoder(in_seq_len, out_seq_len,
                               in_features, mid_features)
        self.attention = Attention(mid_features, n_head, mid_features)

        self.node_embedding = nn.Parameter(
            torch.FloatTensor(num_node, embed_features))
        self.adj = torch.from_numpy(adj).to(device)

        self.dynamic_gcn = GCN(
            mid_features, mid_features, cheb_k, dropout)

        self.static_gcn = GCN(
            mid_features, mid_features, cheb_k, dropout)

        self.downsample = nn.Conv2d(
            2 * mid_features, mid_features, kernel_size=(1, 1))

        self.linear = nn.Linear(mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

        self.reset_parameter()

    def reset_parameter(self):
        for param in self.linear.parameters():
            nn.init.normal_(param)

        for param in self.predict.parameters():
            nn.init.normal_(param)

        self.node_embedding.data.normal_()

    def forward(self, inputs):
        out = self.encoder(inputs)

        attention_res = self.attention(out)

        pre_gcn = out
        static_out = self.static_gcn(out, self.adj)

        dynamic_adj = torch.softmax(torch.relu(torch.matmul(
            self.node_embedding, self.node_embedding.T)), dim=1)
        dynamic_out = self.dynamic_gcn(out, dynamic_adj)

        g_out = torch.concat([static_out, dynamic_out], dim=-1)

        g_out = g_out.permute(0, 3, 2, 1)
        out = self.downsample(g_out)
        out = out.permute(0, 3, 2, 1)

        mix = torch.tanh(pre_gcn) + torch.sigmoid(attention_res)

        out = out + mix

        out = torch.relu(self.linear(out))

        out = self.predict(out)

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 1)
    adj = np.random.randn(307, 307).astype(np.float32)
    net = Archer(307, 12, 12, 1, 32, 1, adj, device='cpu')
    res = net(data)
    print(res.shape)
