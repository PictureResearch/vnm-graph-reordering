import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, HeterophilousGraphDataset, AttributedGraphDataset
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.base import DGLError
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock
from dgl.transforms import reverse
from dgl.utils import expand_as_pair


class EdgeWeightNorm(torch.nn.Module):
    def __init__(self, norm="both", eps=0.0):
        super(EdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):
        
        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError(
                    "Currently the normalization is only defined "
                    "on scalar edge weight. Please customize the "
                    "normalization for your high-dimensional weights."
                )
            if self._norm == "both" and torch.any(edge_weight <= 0).item():
                raise DGLError(
                    'Non-positive edge weight detected with `norm="both"`. '
                    "This leads to square root of zero or negative values."
                )

            dev = graph.device
            dtype = edge_weight.dtype
            graph.srcdata["_src_out_w"] = torch.ones(
                graph.number_of_src_nodes(), dtype=dtype, device=dev
            )
            graph.dstdata["_dst_in_w"] = torch.ones(
                graph.number_of_dst_nodes(), dtype=dtype, device=dev
            )
            graph.edata["_edge_w"] = edge_weight

            if self._norm == "both":
                reversed_g = reverse(graph)
                reversed_g.edata["_edge_w"] = edge_weight
                reversed_g.update_all(
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "out_weight")
                )
                degs = reversed_g.dstdata["out_weight"] + self._eps
                norm = torch.pow(degs, -0.5)
                graph.srcdata["_src_out_w"] = norm

            if self._norm != "none":
                graph.update_all(
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "in_weight")
                )
                degs = graph.dstdata["in_weight"] + self._eps
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata["_dst_in_w"] = norm

            graph.apply_edges(
                lambda e: {
                    "_norm_edge_weights": e.src["_src_out_w"]
                    * e.dst["_dst_in_w"]
                    * e.data["_edge_w"]
                }
            )
            return graph.edata["_norm_edge_weights"]

# pylint: disable=W0235
class GraphConv(torch.nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GraphConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
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
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            # if self._in_feats > self._out_feats:
            if True:
                # mult W first to reduce the feature size for aggregation.
                start_event.record()
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                    
                end_event.record()
                end_event.synchronize()
                lin_time = start_event.elapsed_time(end_event)
                graph.srcdata["h"] = feat_src
                start_event.record()
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                end_event.record()
                end_event.synchronize()
                prop_time = start_event.elapsed_time(end_event)
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                start_event.record()
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                end_event.record()
                end_event.synchronize()
                prop_time = start_event.elapsed_time(end_event)
                rst = graph.dstdata["h"]
                start_event.record()
                if weight is not None:
                    rst = torch.matmul(rst, weight)
                end_event.record()
                end_event.synchronize()
                lin_time = start_event.elapsed_time(end_event)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst, lin_time, prop_time

    def extra_repr(self):
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, activation=F.relu)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x, lin_time1, prop_time1 = self.conv1(g, features)
        x, lin_time2, prop_time2 = self.conv2(g, x)
        return x, lin_time1, prop_time1, lin_time2, prop_time2

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

print("Dataset, hidden, 1st Linear Transformation Latency, 1st Message Passing Latency: , 2nd Linear Transformation Latency, 2nd Message Passing Latency")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup = True
num_hidden = 512

for name, dataset in datasets.items():
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    # Instantiate the model for this dataset
    model = GCN(in_feats=num_features, hidden_size=num_hidden, num_classes=num_classes).to(device)

    # Get a single graph object from each dataset
    data = dataset[0]
    data = convert_pyg_to_dgl(data)
    data = dgl.add_self_loop(data)

    # Measure latency
    model.eval()
    data = data.to(device)
    if warmup:
        model(data, data.ndata['feat'])
        warmup = False
    with torch.no_grad():
        import nvtx
        with nvtx.annotate("start of region", color="blue"):
            _, lin_time1, prop_time1, lin_time2, prop_time2 = model(data, data.ndata['feat'])
        print(f"{name}, {num_hidden}, {lin_time1}, {prop_time1}, {lin_time2}, {prop_time2}")

