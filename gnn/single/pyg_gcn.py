import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, HeterophilousGraphDataset, AttributedGraphDataset

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch

# Define the GCNConv with timing
class TimedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 self_loops=True, normalize=True, bias=True, **kwargs):
        super(TimedGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.self_loops = self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # Timing starts here
        start_lin = torch.cuda.Event(enable_timing=True)
        end_lin = torch.cuda.Event(enable_timing=True)
        start_propagate = torch.cuda.Event(enable_timing=True)
        end_propagate = torch.cuda.Event(enable_timing=True)

        # Normalize edge_index and edge_weight if necessary
        # if self.normalize:
        #     edge_index, edge_weight = self.norm(edge_index, x.size(0), edge_weight,
        #                                         self.improved, self.self_loops)

        # Perform linear transformation and record the time
        start_lin.record()
        x = self.lin(x)
        end_lin.record()

        # Perform message passing and record the time
        start_propagate.record()
        out = self.propagate(edge_index, size=None, x=x, edge_weight=edge_weight)
        end_propagate.record()

        torch.cuda.synchronize()  # Wait for the events to be recorded

        lin_time_ms = start_lin.elapsed_time(end_lin)
        propagate_time_ms = start_propagate.elapsed_time(end_propagate)

        if self.bias is not None:
            out += self.bias

        return out, lin_time_ms, propagate_time_ms

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    # @staticmethod
    # def norm(edge_index, num_nodes, edge_weight, improved, self_loops):
    #     if self_loops:
    #         edge_index, edge_weight = add_self_loops(
    #             edge_index, edge_weight, fill_value=1 if not improved else 2, num_nodes=num_nodes)

    #     row, col = edge_index
    #     deg = degree(col, num_nodes, dtype=edge_weight.dtype)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    #     return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# Define a GCN model that uses the TimedGCNConv
class TimedGCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(TimedGCN, self).__init__()
        self.conv1 = TimedGCNConv(num_features, num_hidden)
        self.conv2 = TimedGCNConv(num_hidden, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x, lin_time1, prop_time1 = self.conv1(x, edge_index)
        x = torch.relu(x)
        x, lin_time2, prop_time2 = self.conv2(x, edge_index)

        total_lin_time = lin_time1 + lin_time2
        total_prop_time = prop_time1 + prop_time2

        return x, [lin_time1, prop_time1, lin_time2, prop_time2]

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

print("Dataset, hidden, 1st Linear Transformation Latency, 1st Message Passing Latency, 2nd Linear Transformation Latency, 2nd Message Passing Latency")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True

num_hidden = 512
for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    # Instantiate the model for this dataset
    model = TimedGCN(num_features=num_features, num_hidden=num_hidden, num_classes=num_classes).to(device)

    # Get a single graph object from each dataset
    data = dataset[0].to(device)

    # Measure latency
    model.eval()
    if warmup:
        model(data)
        warmup = False
    with torch.no_grad():
        _, lin_time1, prop_time1, lin_time2, prop_time2 = model(data)
        print(f"{name}, {num_hidden}, {lin_time1}, {prop_time1}, {lin_time2}, {prop_time2}")