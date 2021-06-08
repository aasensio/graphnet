import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean
import networkx as nx

def visualize(h, color, epoch=None, loss=None):
    pl.figure(figsize=(7,7))
    pl.xticks([])
    pl.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        pl.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            pl.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    pl.show()

def kaiming_init(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class GraphIndependent(nn.Module):
    def __init__(self, edge_model_fn=None, node_model_fn=None, global_model_fn=None):
        """
        A graph block that applies models to the graph elements independently.
        The inputs and outputs are nodes, edges and global. The corresponding models are applied to
        each element of the graph (edges, nodes and globals) in parallel and
        independently of the other elements. It can be used to encode or
        decode the elements of a graph.
        """
        super(GraphIndependent, self).__init__()
                        
        if (edge_model_fn is None):
            self.edge_model_fn = lambda x: x
        else:
            self.edge_model_fn = edge_model_fn

        if (node_model_fn is None):
            self.node_model_fn = lambda x: x
        else:
            self.node_model_fn = node_model_fn

        if (global_model_fn is None):
            self.global_model_fn = lambda x: x
        else:
            self.global_model_fn = global_model_fn
                
    def forward(self, nodes=None, edges=None, glob=None):
        
        breakpoint()
        nodes_out = self.node_model_fn(nodes)
        edges_out = self.edge_model_fn(edges)
        global_out = self.global_model_fn(glob)

        return nodes_out, edges_out, global_out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layers, output_size, activation='relu', layernorm=False):
        """
        
        """
        super(MLP, self).__init__()
                        
        if (activation == 'relu'):
            self.activation = nn.ReLU()
        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(self.activation)

        for i in range(n_hidden_layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.activation)

        self.layers.append(nn.Linear(hidden_size, output_size))
        if (layernorm):
            self.layers.append(nn.LayerNorm(output_size))
                
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class EdgeModel(torch.nn.Module):
    def __init__(self, node_latent_size, edge_latent_size, mlp_hidden_size, mlp_n_hidden_layers, latent_size, activation='relu', layernorm=True):
        super(EdgeModel, self).__init__()

        self.edge_mlp = MLP(
            2*node_latent_size + edge_latent_size,
            mlp_hidden_size, 
            mlp_n_hidden_layers, 
            latent_size, 
            activation=activation, 
            layernorm=layernorm)

    def forward(self, src, dest, edge_attr, u, batch):        
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_latent_size, edge_latent_size, mlp_hidden_size, mlp_n_hidden_layers, latent_size, activation='relu', layernorm=True):
        super(NodeModel, self).__init__()

        self.node_mlp_1 = MLP(
            node_latent_size + edge_latent_size,
            mlp_hidden_size, 
            mlp_n_hidden_layers, 
            latent_size, 
            activation=activation, 
            layernorm=layernorm)

        self.node_mlp_2 = MLP(
            node_latent_size + latent_size,
            mlp_hidden_size, 
            mlp_n_hidden_layers, 
            latent_size, 
            activation=activation, 
            layernorm=layernorm)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class EncodeProcessDecode(nn.Module):
    def __init__(self, node_input_size, edge_input_size, latent_size, mlp_hidden_size, mlp_n_hidden_layers, n_message_passing_steps, output_size):
        """
            node_input_size: Size of the node input representations.
            edge_input_size: Size of the edge input representations.
            latent_size: Size of the node and edge latent representations.
            mlp_hidden_size: Hidden layer size for all MLPs.
            mlp_n_hidden_layers: Number of hidden layers in all MLPs.
            n_message_passing_steps: Number of message passing steps.
            output_size: Output size of the decode node representations
        """
        super(EncodeProcessDecode, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.latent_size = latent_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_n_hidden_layers = mlp_n_hidden_layers
        self.n_message_passing_steps = n_message_passing_steps
        self.output_size = output_size

        # ---------------------
        # ENCODER
        # ---------------------
        self.edge_encoder = MLP(
            self.edge_input_size, 
            self.mlp_hidden_size, 
            self.mlp_n_hidden_layers, 
            self.latent_size, 
            activation='relu', 
            layernorm=True)

        self.node_encoder = MLP(
            self.node_input_size, 
            self.mlp_hidden_size, 
            self.mlp_n_hidden_layers, 
            self.latent_size, 
            activation='relu', 
            layernorm=True)

        self.encoder_network = GraphIndependent(edge_model_fn=self.edge_encoder, node_model_fn=self.node_encoder)


        # ---------------------
        # DECODER
        # ---------------------
        self.decoder_network = MLP(
            self.latent_size, 
            self.mlp_hidden_size, 
            self.mlp_n_hidden_layers, 
            self.output_size,
            activation='relu', 
            layernorm=False)

        # ---------------------
        # PROCESSOR
        # ---------------------
        self.processor_network = nn.ModuleList([])
        for i in range(self.n_message_passing_steps):
            edge_model_fn = EdgeModel(                
                node_latent_size=self.latent_size, 
                edge_latent_size=self.latent_size,
                mlp_hidden_size=self.mlp_hidden_size, 
                mlp_n_hidden_layers=self.mlp_n_hidden_layers, 
                latent_size=self.latent_size, 
                activation='relu',
                layernorm=True)

            node_model_fn = NodeModel(                
                node_latent_size=self.latent_size, 
                edge_latent_size=self.latent_size,
                mlp_hidden_size=self.mlp_hidden_size, 
                mlp_n_hidden_layers=self.mlp_n_hidden_layers, 
                latent_size=self.latent_size, 
                activation='relu', 
                layernorm=True)
            
            self.processor_network.append(MetaLayer(edge_model_fn, node_model_fn, None))

    def encode(self, node, edge_attr):
        breakpoint()
        node_out, edge_attr_out, _ = self.encoder_network(
            nodes=node,
            edges=edge_attr)

        return node_out, edge_attr_out

    def decode(self, node):
        node_out = self.decoder_network(node)

        return node_out

    def process_step(self, processor_network, node, edge_attr, edge_index):
        
        # One message passing
        node_k, edge_attr_k, _ = processor_network(node, edge_index, edge_attr, None, None)

        # Add residuals
        node_k = node_k + node
        edge_attr_k = edge_attr_k + edge_attr

        return node_k, edge_attr_k

    def process(self, node, edge_attr, edge_index):
        
        # Do all messsage passing steps         
        node_prev_k = node
        edge_attr_prev_k = edge_attr
        
        for i in range(self.n_message_passing_steps):
            node_k, edge_attr_k = self.process_step(self.processor_network[i], node_prev_k, edge_attr_prev_k, edge_index)
            
            node_prev_k = node_k
            edge_attr_prev_k = edge_attr_k

        return node_k, edge_attr_k

    def forward(self, node, edge_attr, edge_index):

        # Encode to latent space
        node_0, edge_attr_0 = self.encode(node, edge_attr)

        # Process all steps
        node_k, edge_attr_k = self.process(node_0, edge_attr_0, edge_index)                

        # Decode output
        node_out = self.decode(node_k)    

        return node_out            

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)


if (__name__ == '__main__'):
    n_node = 20
    n_node_attr = 1
    n_edge_attr = 1
    n_edges = 2*(n_node - 2) + 2

    node = torch.zeros((n_node, n_node_attr), dtype=torch.float)
    for i in range(n_node):
        node[i, :] = i / 10.0
    target = torch.zeros((n_node, n_node_attr), dtype=torch.float)
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    edge_attr = torch.zeros((n_edges, n_edge_attr), dtype=torch.float)

    loop = 0
    for i in range(n_node):
        if (i == 0):
            edge_index[0, loop] = 0
            edge_index[1, loop] = 1
            edge_attr[loop, :] = torch.ones(1)
            loop += 1
        if (i == n_node-1):
            edge_index[0, loop] = n_node-1
            edge_index[1, loop] = n_node-2
            edge_attr[loop, :] = torch.ones(1)
            loop += 1
        if (i > 0 and i < n_node-1):
            edge_index[0, loop] = i
            edge_index[1, loop] = i-1
            edge_attr[loop, :] = torch.ones(1)
            loop += 1
            edge_index[0, loop] = i
            edge_index[1, loop] = i+1
            edge_attr[loop, :] = torch.ones(1)
            loop += 1
    
    data = Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=target)

    # G = to_networkx(data, to_undirected=True)
    # visualize(G, color=data.y)

    net = EncodeProcessDecode(
        node_input_size=1, 
        edge_input_size=1, 
        latent_size=16, 
        mlp_hidden_size=16, 
        mlp_n_hidden_layers=2, 
        n_message_passing_steps=2,
        output_size=1)
    
    tmp = net(node, edge_attr, edge_index)