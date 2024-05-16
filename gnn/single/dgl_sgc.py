from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, HeterophilousGraphDataset, AttributedGraphDataset
from torch_geometric.data import Data

from typing import Optional
import torch.nn.functional as F
import torch

from typing import Optional

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm, subgraph

class SGConv(torch.nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=None,
        allow_zero_in_degree=False,
    ):
        super(SGConv, self).__init__()
        self.fc = torch.nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        time_record = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    if edge_weight is None:
                        feat = feat * norm
                    graph.ndata["h"] = feat
                    start_event.record()
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    end_event.record()
                    end_event.synchronize()
                    time_record.append(start_event.elapsed_time(end_event))
                    feat = graph.ndata.pop("h")
                    if edge_weight is None:
                        feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            start_event.record()
            x = self.fc(feat)
            end_event.record()
            end_event.synchronize()
            time_record.append(start_event.elapsed_time(end_event))
            return x, time_record

class SGCNet(torch.nn.Module):
    def __init__(self, in_feats, out_feats, k, cached=False, bias=True):
        super(SGCNet, self).__init__()
        self.sgc = SGConv(in_feats, out_feats, k=k, cached=cached, bias=bias)

    def forward(self, g, features):
        h, time_record = self.sgc(g, features)
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

print("Dataset, hidden, 1st Message Passing Latency, 2nd Message Passing Latency, 1st Linear Latency")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True
hidden_size = 512

for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes 

    # Initialize the model
    # model = SGCNet(num_features, num_classes).to(device)
    # num_features = hidden_size # args.hidden
    model = SGCNet(num_features, num_classes, k=2).to(device)

    # Get a single graph object from each dataset
    data = dataset[0]

    data = convert_pyg_to_dgl(data)
    data = dgl.add_self_loop(data)

    # Measure latency
    model.eval()
    data = data.to(device)
    #print(data.edge_index)
    if warmup:
        model(data, data.ndata['feat'])
        warmup = False
    with torch.no_grad():
        _ , time_record= model(data, data.ndata['feat'])
        print(f"{name}, " + ", ".join([str(t) for t in time_record]))