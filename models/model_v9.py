from importlib.util import resolve_name
from unittest import result
import torch
import torch.nn as nn
import numpy as np

class TCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 3) -> None:
        super(TCNLayer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, 1))
            for _ in range(3)
        ])  # P&Q and reserved

        self.bn = nn.BatchNorm2d(out_dim)
        self.dropout = 0.5
    
    def forward(self, inputs):
        """
        param inputs: (batch_size, num_timestamp, num_node, in_dim)
        return: (batch_size, num_timestamp - 2, num_node, out_dim)
        """
        inputs = inputs.permute(0, 3, 1, 2)
        out = torch.tanh(self.convs[0](inputs)) + \
            torch.sigmoid(self.convs[1](inputs))
        # out = self.convs[2](inputs) + out

        out = torch.dropout(self.bn(out), p=self.dropout, train=self.training)
        out = out.permute(0, 2, 3, 1)
        return out


class TCN(nn.Module):
    def __init__(self, n_series_len, in_dim, out_dim) -> None:
        super(TCN, self).__init__()
        # odd time seriers: number of layer is n_series_len // 2
        # even time seriers: number of layer is n_series_len // 2 - 1 + a single conv layer
        # -> Aggreate information from time seriers to one unit
        assert n_series_len >= 3
        self.is_even = False if n_series_len % 2 != 0 else True

        self.n_layer = n_series_len // 2 if n_series_len % 2 != 0 else (n_series_len // 2 - 1)

        self.layers = nn.ModuleList([TCNLayer(in_dim, out_dim)])
        for _ in range(self.n_layer - 1):
            self.layers.append(TCNLayer(out_dim, out_dim))
        
        if self.is_even:
            self.layers.append(TCNLayer(out_dim, out_dim, kernel_size=2))

        self.upsample = None if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, kernel_size=1
        )

    def forward(self, inputs):
        out = self.layers[0](inputs)
        if self.upsample:
            ResConn = self.upsample(inputs.permute(0, 3, 2, 1))
            ResConn = ResConn.permute(0, 3, 2, 1)
        else:
            ResConn = inputs
        
        out = out + ResConn[:, 2:, ...]

        for i in range(1, self.n_layer):
            out = self.layers[i](out) + out[:, 2:, ...] + ResConn[:, 2 * (i + 1):, ...]
        
        if self.is_even:
            out = self.layers[-1](out) + out[:,-1:,...] + ResConn[:,-1:,...]
        
        return out
    

class Encoder(nn.Module):
    def __init__(self, n_history, n_predict, in_dim, out_dim) -> None:
        super(Encoder, self).__init__()
        assert n_history >= 3

        self.n_predict = n_predict

        self.tcn = TCN(n_history, in_dim, out_dim)

        self.fully = nn.Linear(out_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        for param in self.fully.parameters():
            param.data.normal_()
    
    def forward(self, inputs):
        """
        inputs: (batch_size, n_history, num_nodes, in_dim)
        return: (batch_size, 1, num_nodes, out_dim)
        """
        out = self.tcn(inputs)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.fully(torch.relu(out))
        out = out.reshape(out.shape[0], out.shape[1], 1, -1)
        out = out.permute(0, 2, 1, 3)
        return out


class GraphCN(nn.Module):
    def __init__(self, in_dim, out_dim, cheb_k=3, dropout=0.5) -> None:
        super(GraphCN, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.cheb_k = cheb_k
        self.dropout = dropout

        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamps, num_node, num_features)
        adj: (num_node, num_node)
        """
        supports = [torch.eye(adj.shape[0]).to(adj.device), adj]
        for k in range(2, self.cheb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])
        
        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports, inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bias)

        return result.permute(0, 2, 1, 3)


class TransLayer(nn.Module):
    def __init__(self, num_node, n_history, n_predict, in_dim, n_head, n_dim_per_head) -> None:
        super(TransLayer, self).__init__()
        D = n_head * n_dim_per_head
        
        self.num_node = num_node
        self.n_predict = n_predict
        self.n_history = n_history
        self.n_head = n_head
        self.n_dim_per_head = n_dim_per_head

        self.inLinear = nn.Linear(1, n_predict)
        
        self.convQ = nn.Conv2d(in_dim, D, kernel_size=(1, 1))
        self.convK = nn.Conv2d(in_dim, D, kernel_size=(1, 1))
        self.convV = nn.Conv2d(in_dim, D, kernel_size=(1, 1))

        self.lengthLinear = nn.Linear(n_history, n_predict)
        self.linear = nn.Linear(D, n_dim_per_head)

        self.LayerNorm = nn.LayerNorm([n_dim_per_head, n_predict, num_node])

    
    def forward(self, inputs, findSeries):
        """
        inputs: (batch_size, 1, num_node, in_features)
        findSeries: (batch_size, n_history, num_node, in_dim)
        """

        batch_size = inputs.shape[0]

        # (batch_size, in_dim, num_node, 1)
        inputs = inputs.permute(0, 3, 2, 1)
        # (batch_size, in_dim, num_node, n_predict)
        out = self.inLinear(inputs)

        ResConn = out[:]

        # (batch_size, in_dim, num_node, n_history)
        findSeries = findSeries.permute(0, 3, 2, 1)

        # (batch_size, n_head * n_dim_per_head, num_node, n_history)
        query = self.convQ(findSeries)
        # (batch_size, n_head * n_dim_per_head, num_node, n_predict)
        key = self.convK(out)
        # (batch_size, n_head * n_dim_per_head, num_node, n_predict)
        value = self.convV(out)


        # (n_head * batch_size, n_dim_per_head, num_node, timestamp)
        query = torch.concat(torch.split(query, self.n_dim_per_head, dim=1), dim=0)
        key = torch.concat(torch.split(key, self.n_dim_per_head, dim=1), dim=0)
        value = torch.concat(torch.split(value, self.n_dim_per_head, dim=1), dim=0)

        # (n_head * batch_size, num_node, n_history, n_dim_per_head)
        query = query.permute(0, 2, 3, 1)
        # (n_head * batch_size, num_node, n_dim_per_head, n_predict)
        key = key.permute(0, 2, 1, 3)
        # (n_head * batch_size, num_node, n_predict, n_dim_per_head)
        value = value.permute(0, 2, 3, 1)

        attention = torch.matmul(query, key)
        attention = attention / self.n_dim_per_head ** 0.5
        attention = torch.softmax(attention, dim=-1)

        result = torch.matmul(attention, value)
        result = torch.concat(torch.split(result, batch_size, dim=0), dim=-1)
        result = torch.relu(self.linear(result)) # (batch_size, num_node, n_history, n_dim_per_head)

        if self.n_history != self.n_predict:
            result = result.permute(0, 1, 3, 2)
            result = self.lengthLinear(result)
            result = result.permute(0, 1, 3, 2)
        
        # (batch_size, n_dim_per_head, num_node, n_predict)
        result = result.permute(0, 3, 1, 2)
        result = result + ResConn
        result = result.permute(0, 1, 3, 2)

        # (batch_size, n_dim_per_head, n_predict, num_node)
        result = self.LayerNorm(result)

        return result.permute(0, 2, 3, 1)


class MixGraphResult(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.stConv = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1))
        self.dyConv = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1))
        self.act = nn.LeakyReLU()
        
        self.reset_parameter()
        pass

    def reset_parameter(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

    def forward(self, stResult, dyResult):
        """
        stResult: (batch_size, timestamp, num_node, in_dim)
        dyResult: (batch_size, timestamp, num_node, in_dim)
        return: (batch_size)
        """
        # (batch_size, in_dim, timestamp, num_node)
        stResult = stResult.permute(0, 3, 1, 2)
        dyResult = dyResult.permute(0, 3, 1, 2)

        # (batch_size, out_dim, timestamp, num_node)
        sOut = self.stConv(stResult)
        dOut = self.dyConv(dyResult)

        out = self.act(sOut) + torch.sigmoid(dOut)
        out += dOut
        return out.permute(0, 2, 3, 1)




class Archer(nn.Module):
    def __init__(self, num_node, num_embed, n_history, n_predict, in_dim, mid_dim, out_dim, adj) -> None:
        super(Archer, self).__init__()

        self.preLinear = nn.Linear(in_dim, mid_dim)

        self.adj = nn.Parameter(torch.from_numpy(adj), requires_grad=False)
        self.embed = nn.Parameter(torch.FloatTensor(num_node, num_embed))

        self.encoder = Encoder(n_history, n_predict, mid_dim, mid_dim)

        self.trans = TransLayer(num_node, n_history, n_predict, mid_dim, n_head=4, n_dim_per_head=mid_dim)

        self.gcn = GraphCN(mid_dim, mid_dim)
        self.gcn_with_embed = GraphCN(mid_dim, mid_dim)

        self.act = nn.LeakyReLU()

        self.downsample = nn.Conv2d(
            2 * mid_dim, mid_dim, kernel_size=1)
        self.mixGraphRes = MixGraphResult(mid_dim, mid_dim)
        
        self.fully = nn.Linear(mid_dim, mid_dim)
        self.predict = nn.Linear(mid_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        self.embed.data.normal_()

    
    def forward(self, inputs):
        """
        inputs: (batch_size, n_history, num_node, in_dim)
        return: (batch_size, n_predict, num_node, out_dim)
        """

        out = torch.relu(self.preLinear(inputs))
        inputs = out[:]

        out = self.encoder(out)
        out = self.trans(out, inputs)

        ResConn = out[:]

        dynamic_adj = torch.softmax(self.act(torch.matmul(
            self.embed, self.embed.T
        )), dim=1)
        embed_out = self.gcn_with_embed(out, dynamic_adj)

        out = self.gcn(out, self.adj)
        out = self.mixGraphRes(out, embed_out)

        # out = torch.cat([embed_out, out], dim=-1)

        # out = out.permute(0, 3, 2, 1)
        # out = self.downsample(out)
        # out = out.permute(0, 3, 2, 1)
        # print(out.shape)

        out = out + ResConn

        out = self.act(self.fully(out))
        out = self.predict(out)

        return out


if __name__ == '__main__':
    data = torch.randn(64, 12, 307, 1)
    adj = np.random.rand(307, 307).astype(np.float32)
    net = Archer(307, 9, 12, 12, 1, 32, 1, adj)
    net(data)

