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

class SGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        time_record = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                start_event.record()
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                end_event.record()
                end_event.synchronize()
                time_record.append(start_event.elapsed_time(end_event))
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        start_event.record()
        x = self.lin(x)
        end_event.record()
        end_event.synchronize()
        time_record.append(start_event.elapsed_time(end_event))
        return x, time_record

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


class SGCNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SGCNet, self).__init__()
        self.conv = SGConv(num_node_features, num_classes, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x, time_record = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1), time_record

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

print("Dataset, hidden, 1st Msg Pass, 2nd Msg Pass, linear")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True

for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Initialize the GraphSAGE model
    model = SGCNet(num_features, num_classes).to(device)

    # Get a single graph object from each dataset
    data = dataset[0].to(device)

    # Measure latency
    model.eval()
    data = data.to(device)
    #print(data.edge_index)

    if warmup:
        model(data)
        warmup = False
    with torch.no_grad():
        _ , time_record= model(data)
        print(f"{name}, " + ", ".join([str(t) for t in time_record]))