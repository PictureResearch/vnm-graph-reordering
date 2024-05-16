from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, HeterophilousGraphDataset, AttributedGraphDataset
from torch_geometric.data import Data

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, NeighborSampler

import dgl
from dgl import broadcast_nodes, function as fn
from dgl.base import dgl_warning

class ChebConv(torch.nn.Module):
    def __init__(self, in_feats, out_feats, k, activation=F.relu, bias=True):
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = torch.nn.Linear(k * in_feats, out_feats, bias)

    def forward(self, graph, feat, lambda_max=None):

        def unnLaplacian(feat, D_invsqrt, graph):
            """Operation Feat * D^-1/2 A D^-1/2"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return graph.ndata.pop("h") * D_invsqrt
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        time_record = []
        with graph.local_scope():
            D_invsqrt = torch.pow(
                graph.in_degrees().to(feat).clamp(min=1), -0.5
            ).unsqueeze(-1)

            if lambda_max is None:
                dgl_warning(
                    "lambda_max is not provided, using default value of 2.  "
                    "Please use dgl.laplacian_lambda_max to compute the eigenvalues."
                )
                lambda_max = [2] * graph.batch_size

            if isinstance(lambda_max, list):
                lambda_max = torch.Tensor(lambda_max).to(feat)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            
            lambda_max = broadcast_nodes(graph, lambda_max)
            re_norm = 2.0 / lambda_max

            # X_0 is the raw feature, Xt is the list of X_0, X_1, ... X_t
            X_0 = feat
            Xt = [X_0]

            # X_1(f)
            if self._k > 1:
                start_event.record()
                h = unnLaplacian(X_0, D_invsqrt, graph)
                end_event.record()
                end_event.synchronize()
                time_record.append(start_event.elapsed_time(end_event))
                X_1 = -re_norm * h + X_0 * (re_norm - 1)
                # Append X_1 to Xt
                Xt.append(X_1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                start_event.record()
                h = unnLaplacian(X_1, D_invsqrt, graph)
                end_event.record()
                end_event.synchronize()
                time_record.append(start_event.elapsed_time(end_event))
                X_i = -2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0
                # Add X_1 to Xt
                Xt.append(X_i)
                X_1, X_0 = X_i, X_1

            # Create the concatenation
            Xt = torch.cat(Xt, dim=1)

            # linear projection
            start_event.record()
            h = self.linear(Xt)
            end_event.record()
            end_event.synchronize()
            time_record.append(start_event.elapsed_time(end_event))
            # activation
            if self.activation:
                h = self.activation(h)

        return h, time_record

class ChebNet(torch.nn.Module):
    def __init__(self, in_feats, out_feats, K, bias=True):
        super(ChebNet, self).__init__()
        self.cheb_conv = ChebConv(in_feats, out_feats, K, bias=bias)

    def forward(self, graph, features):
        # Apply ChebConv
        h, time_record = self.cheb_conv(graph, features)
        return h, time_record

# Helper function to combine graphs
def combine_graphs(dataset):
    # Initialize combined lists
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []

    # Edge index offset will keep track of the number of nodes so far
    edge_index_offset = 0
    for data in dataset:
        # Append node features and labels
        x_list.append(data.x)
        y_list.append(data.y)

        # Offset edge indices and append them
        edge_index_list.append(data.edge_index + edge_index_offset)

        # If dataset provides edge attributes, append them
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)

        # Update the edge index offset
        edge_index_offset += data.num_nodes

    # Concatenate everything to form a single graph
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    y = torch.cat(y_list, dim=0)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None

    # Create a single Data object
    combined_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

    return combined_data

def convert_pyg_to_dgl(pyg_data):
    # pyg_data is an instance of torch_geometric.data.Data

    # Convert the graph structure
    # PyG's edge_index is of shape [2, num_edges]
    src, dst = pyg_data.edge_index
    dgl_graph = dgl.graph((src, dst))

    # Transfer node features
    if pyg_data.x is not None:
        dgl_graph.ndata['feat'] = pyg_data.x

    # Transfer node labels (if any)
    if hasattr(pyg_data, 'y'):
        dgl_graph.ndata['label'] = pyg_data.y

    return dgl_graph

# Load datasets and measure latency
datasets = {
    'Cora': Planetoid(root='./dataset/Cora', name='Cora'),
    'Citeseer': Planetoid(root='./dataset/Citeseer', name='Citeseer'),
    'Facebook':AttributedGraphDataset(root='./dataset/Facebook', name='Facebook'),
    'Computers':Amazon(root='./dataset/Computers', name='Computers'),
    'CS':Coauthor(root='./dataset/CS', name='CS'),
    'Cora_Full':CitationFull(root='./dataset/Cora_Full', name='Cora'),
    'Amazon-ratings':HeterophilousGraphDataset(root='./dataset/Amazon-ratings', name='Amazon-ratings'),
    'Physics':Coauthor(root='./dataset/Physics', name='Physics'),
}

print("Dataset, hidden, 1st Message Passing Latency, 2nd Message Passing Latency, 3rd Message Passing Latency, 1st Linear Latency")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True
num_hidden = 512

for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Initialize the model
    model = ChebNet(num_features, num_hidden, 4).to(device)
    
    # Get a single graph object from each dataset
    data = dataset[0].to(device)
    data = convert_pyg_to_dgl(data)
    data = dgl.add_self_loop(data)
    
    # Measure latency
    model.eval()
    data = data.to(device)
    if warmup:
        model(data, data.ndata['feat'])
        warmup = False
    with torch.no_grad():
        _ , time_record= model(data, data.ndata['feat'])
        print(f"{name}, {num_hidden}, " + ", ".join([str(t) for t in time_record]))

