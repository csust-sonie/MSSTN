from unittest import result
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, num_node, n_history, in_features, n_head) -> None:
        super(SpatialAttention, self).__init__()

        D = n_head * in_features
        self.in_features = in_features

        self.ConvQ = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.ConvK = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.ConvV = nn.Conv2d(in_features, D, kernel_size=(1, 1))

        self.multi_head_confusion = nn.Conv2d(
            D, in_features, kernel_size=(1, 1))

        self.LayerNorm = nn.LayerNorm([n_history, num_node, in_features])

    def forward(self, inputs):
        """
        param inputs: (batch_size, n_history, num_node, in_features)
        """
        batch_size = inputs.shape[0]
        num_node = inputs.shape[2]
        inputs = inputs.permute(0, 3, 2, 1)  # (btnf)->(bfnt)

        # (bs, in_feature * n_head, num_node, n_time)
        Q = self.ConvQ(inputs)
        K = self.ConvK(inputs)
        V = self.ConvV(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        # (bs * n_head, in_features, num_node, n_time)
        Q = torch.cat(torch.split(
            Q, split_size_or_sections=self.in_features, dim=1), dim=0)
        K = torch.cat(torch.split(
            K, split_size_or_sections=self.in_features, dim=1), dim=0)
        V = torch.cat(torch.split(
            V, split_size_or_sections=self.in_features, dim=1), dim=0)

        K = K.permute(0, 3, 2, 1)  # (btnf)
        Q = Q.permute(0, 3, 1, 2)  # (btfn)
        V = V.permute(0, 3, 2, 1)  # (btnf)

        attention = (K @ Q) / (num_node ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        out = attention @ V
        # 我们通过空间注意力机制？就是加权找一下节点依存关系
        # 编码又回到了self_attention上

        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = out.permute(0, 3, 2, 1)

        # (bftn)
        out = self.multi_head_confusion(out)

        out = out.permute(0, 3, 2, 1)

        out += inputs

        out = self.LayerNorm(out)

        return out


class GradualStackGraph(nn.Module):
    def __init__(self, n_history) -> None:
        super(GradualStackGraph, self).__init__()
        self.trans = nn.Linear(n_history, 1)
        self.act = nn.ReLU()

    def forward(self, inputs):
        """
        param: inputs: (batch_size, n_history, num_node, in_features)
        """
        inputs = inputs.permute(0, 3, 2, 1)
        out = self.act(self.trans(inputs)).permute(0, 3, 2, 1).squeeze(1)

        return out @ out.permute(0, 2, 1)


class GCNCell(nn.Module):
    def __init__(self, batch_size, num_node, in_features, out_features, cheb_k=3) -> None:
        super(GCNCell, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(
            batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.cheb_k = cheb_k

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, inputs, adj):
        """
        param: (batch_size, n_history, num_node, in_features)
        """
        eyes = torch.stack([torch.eye(adj.shape[1], adj.shape[1])
                           for i in range(adj.shape[0])], dim=0).to(adj.device)
        supports = [eyes, adj]

        for _ in range(2, self.cheb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])

        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kbij, jbtf->btif', supports,
                           inputs.permute(2, 0, 1, 3))

        result = torch.relu(torch.einsum(
            "btif,bfo->btio", lfs, self.weight) + self.bias)

        return result


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

        self.fully = nn.Linear(mid_features, mid_features)

        self.reset_parameter()

    def reset_parameter(self):
        for param in self.fully.parameters():
            param.data.normal_()

    def forward(self, inputs):
        out = self.tcn(inputs)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.relu(self.fully(out))
        out = out.reshape(out.shape[0], out.shape[1], 1, -1)
        out = out.permute(0, 2, 1, 3)
        return out


class Archer(nn.Module):
    def __init__(self, in_features, mid_features, out_features, num_node, n_history, n_predict, batch_size) -> None:
        super(Archer, self).__init__()
        self.incrDimention = nn.Linear(in_features, mid_features)
        self.DGraphGenerator = SpatialAttention(
            num_node, n_history, mid_features, 4)
        self.Gradual = GradualStackGraph(n_history)
        self.GCN = GCNCell(batch_size, num_node, mid_features, mid_features)

        self.encoder = Encoder(n_history, n_predict,
                               mid_features, mid_features)

        self.transConv = nn.Conv2d(1, n_predict, kernel_size=(1, 1))

        self.linear = nn.Linear(mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

    def forward(self, inputs):
        """
        param: inputs: (batch_size, n_history, num_node, in_features)
        """
        out = self.incrDimention(inputs)  # (btni->btnm)
        res = out[:]

        res = self.encoder(res)

        res = torch.relu(self.transConv(res))

        out = self.DGraphGenerator(out)
        out = self.Gradual(out)
        gcnOut = self.GCN(res, out)

        out = torch.relu(self.linear(gcnOut))
        out = self.predict(out)

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 1)
    # net = Archer(1, 32, 1, 307, 12, 12, 64)

    # res = net(data)

    # print(res.shape)

    data = torch.FloatTensor(data)
    data.detach().cpu().numpy()
