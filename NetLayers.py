import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
display_details=0
#website:https://github.com/chaitjo/graph-convnet-tsp/blob/master/models/gcn_layers.py
# class BatchNormNode(nn.Module):
#     """Batch normalization for node features.
#     """
#     def __init__(self, hidden_dim):
#         super(BatchNormNode, self).__init__()
#         self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
#         if (torch.cuda.is_available()):
#             self.device=torch.device('cuda:0')
#         else:
#             self.device = torch.device('cpu')
#     def forward(self, x):
#         """
#         Args:
#             x: Node features (batch_size, num_nodes, hidden_dim)
#
#         Returns:
#             x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
#         """
#         x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
#         x_trans_bn = self.batch_norm(x_trans)
#         x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
#         return x_bn

#
# class BatchNormEdge(nn.Module):
#     """Batch normalization for edge features.
#     """
#
#     def __init__(self, hidden_dim):
#         super(BatchNormEdge, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
#         if (torch.cuda.is_available()):
#             self.device=torch.device('cuda:0')
#         else:
#             self.device = torch.device('cpu')
#     def forward(self, e):
#         """
#         Args:
#             e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
#
#         Returns:
#             e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
#         """
#         e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
#         print(e_trans)
#         e_trans_bn = self.batch_norm(e_trans)
#         print(e_trans_bn)
#         sys.exit()
#         e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
#         return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.

    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]

    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    def __init__(self, hidden_dim, aggregation="sum"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        self.hidden_dim=hidden_dim
        if (torch.cuda.is_available()):
            self.device=torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    def forward(self, x, edge_gate,adjacency,PN,NN):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)/2  # B x V x H
        Vx = self.V(x)/2  # B x V x H
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        self.hidden_dim=hidden_dim
        if (torch.cuda.is_available()):
            self.device=torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    def forward(self, x, e,PN,NN):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        U = self.U(e)/2 #Edge
        V = self.V(x)/2 #Node
        Wx = V.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = V.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = U/2 + Vx + Wx #add together, The size can be arbitrary. B x V x V x H
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.``
    """
    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        #self.bn_node = BatchNormNode(hidden_dim)
        #self.bn_edge = BatchNormEdge(hidden_dim)
        self.layer_norm_node=nn.LayerNorm(hidden_dim)
        self.layer_norm_edge=nn.LayerNorm(hidden_dim)
        self.leakyrelu=nn.LeakyReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        if (torch.cuda.is_available()):
            self.device=torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    def forward(self, x, e, adjacency, PN,NN):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """

        if display_details: print('x0', x[0, 0, 0:32])
        if display_details: print('x1', x[0, 1, 0:32])
        e_in = e #edge feature
        x_in = x #node feature
        if display_details:print('e0',e[0,0,0,0:32])
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in,PN,NN)  # B x V x V x H convert a second time, H to H dimensions
        if display_details:print('e_tmp0',e_tmp[0,0,0,0:32])
        # Compute edge gates
        #edge_gate = torch.sigmoid(e_tmp) #capture nonlinearity with sigmoid
        edge_gate=self.sigmoid(e_tmp)
        if display_details:print('edgegate', edge_gate[0, 0, 0, 0:32])
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate,adjacency,PN,NN) # convert a second time, H to H dimensions
        if display_details:print('x_tmp0',x_tmp[0,0,0:32])
        e_tmp =self.layer_norm_edge(self.leakyrelu(e_tmp))/2#batch normalisation
        x_tmp = self.layer_norm_node(self.leakyrelu(x_tmp))/2
        if display_details:print('x_tmp1', x_tmp[0, 0, 0:32])
        if display_details:print('e_tmp1', e_tmp[0, 0, 0, 0:32])
        # ReLU Activation
        # Residual connection
        x_new = (x_in/2 + x_tmp)
        e_new = (e_in/2 + e_tmp)

        
      ## sys.exit()
        return x_new, e_new
    

class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """
    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)
        self.leakyrelu = nn.LeakyReLU()
        if (torch.cuda.is_available()):
            self.device=torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)
        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = self.leakyrelu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
