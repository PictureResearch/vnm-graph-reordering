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

class SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = torch.nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = torch.nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = torch.nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = torch.nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            torch.nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            torch.nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            # lin_before_mp = self._in_src_feats > self._out_feats
            lin_before_mp = False

            # Message Passing
            if self._aggre_type == "mean":
                
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                start_event.record()
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                # graph.update_all(msg_fn, fn.sum(msg="m", out="h"))
                end_event.record()
                end_event.synchronize()
                propagate_time = start_event.elapsed_time(end_event)
                h_neigh = graph.dstdata["neigh"]
                # h_neigh = graph.dstdata["h"]
                start_event.record()
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                end_event.record()
                end_event.synchronize()
                lin_l_time = start_event.elapsed_time(end_event)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                start_event.record()
                rst = self.fc_self(h_self) + h_neigh
                end_event.record()
                end_event.synchronize()
                lin_r_time = start_event.elapsed_time(end_event)

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst, propagate_time, lin_l_time, lin_r_time

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, 'mean')

    def forward(self, g, features):
        x, propagate_time_1, lin_l_time_1, lin_r_time_1 = self.conv1(g, features)
        h = F.relu(x)
        h, propagate_time_2, lin_l_time_2, lin_r_time_2 = self.conv2(g, h)
        return h, [propagate_time_1, lin_l_time_1, lin_r_time_1, propagate_time_2, lin_l_time_2, lin_r_time_2]
        
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