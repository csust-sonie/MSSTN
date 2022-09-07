import math
import torch
import torch.nn as nn


class GCNBlock(nn.Module):
    def __init__(self, num_node, in_features, out_features) -> None:
        super(GCNBlock, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))
        self.batch_norm = nn.BatchNorm2d(num_node)

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.data.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bais.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, num_node, num_timestamp, num_features)
        adj: (num_node, num_node)
        """
        lfs = torch.einsum('ij,jbtf->bitf', [adj, inputs.permute(1, 0, 2, 3)])
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bais)
        return self.batch_norm(result)
