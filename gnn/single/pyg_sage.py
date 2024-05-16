import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, HeterophilousGraphDataset, AttributedGraphDataset
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import spmm
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor

class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])
        super().__init__(aggr, **kwargs)
        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)
        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]
        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()
    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        # Start measuring time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)
        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
        # Measure self.propagate
        start_event.record()
        out = self.propagate(edge_index, x=x, size=size)
        end_event.record()
        end_event.synchronize()
        propagate_time = start_event.elapsed_time(end_event)
        # Measure self.lin_l
        start_event.record()
        out = self.lin_l(out)
        end_event.record()
        end_event.synchronize()
        lin_l_time = start_event.elapsed_time(end_event)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            # Measure out + self.lin_r(x_r)
            start_event.record()
            out = out + self.lin_r(x_r)
            end_event.record()
            end_event.synchronize()
            lin_r_time = start_event.elapsed_time(end_event)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out, propagate_time, lin_l_time, lin_r_time
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Define the first GraphSAGE convolution layer
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        # Define the second GraphSAGE convolution layer
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
    def forward(self, data): # x, edge_index
        x, edge_index = data.x, data.edge_index
        # First GraphSAGE layer
        x, propagate_time_1, lin_l_time_1, lin_r_time_1 = self.conv1(x, edge_index)
        x = torch.relu(x)
        # Second GraphSAGE layer
        x, propagate_time_2, lin_l_time_2, lin_r_time_2 = self.conv2(x, edge_index)
        return x, [propagate_time_1, lin_l_time_1, lin_r_time_1, propagate_time_2, lin_l_time_2, lin_r_time_2]
    
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

print("Dataset, hidden, 1st Message Passing Latency, 1st Linear Transformation L Latency, 1st Linear Transformation R Latency, 2nd Message Passing Latency, 2nd Linear Transformation L Latency, 2nd Linear Transformation R Latency")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True

num_hidden = 512
for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Initialize the GraphSAGE model
    model = GraphSAGE(num_features, num_hidden, num_classes).to(device)

    # Get a single graph object from each dataset
    data = dataset[0].to(device)

    # Measure latency
    model.eval()
    if warmup:
        model(data.x, data.edge_index)
        warmup = False
    with torch.no_grad():
        _, propagate_time_1, lin_l_time_1, lin_r_time_1, propagate_time_2, lin_l_time_2, lin_r_time_2 = model(data.x, data.edge_index)
        print(f"{name}, {num_hidden}, {propagate_time_1}, {lin_l_time_1}, {lin_r_time_1}, {propagate_time_2}, {lin_l_time_2}, {lin_r_time_2}")
