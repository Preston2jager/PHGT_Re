import torch_geometric.nn as pyg_nn

class StructureEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SimpleHGNConv(in_channels, hidden_channels, ...)
        self.conv2 = SimpleHGNConv(hidden, hidden)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x
