# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
from utils import PDE

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
    

class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 spatial_dim: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            spatial_dim (int): number of dimension of spatial domain  
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean') # node_dim: The axis along which to propagate. (default: -2)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        assert (spatial_dim == 1 or spatial_dim == 2 or spatial_dim == 3)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + spatial_dim + n_variables, hidden_features), 
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), 
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features), 
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), 
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update



class MPNN(torch.nn.Module):
    """
    Message Passing Neural PDE SOLVERS
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}):
        """
        Initialize MPNN
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE to solve
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MPNN, self).__init__()
        assert(time_window == 25 or time_window == 20 or time_window == 50 or time_window == 10) # add time_window == 10
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        # modified
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            spatial_dim= self.pde.spatial_dim,
            n_variables=1+len(self.eq_variables) # variables = eq_variables + time
            ) for _ in range(self.hidden_layer)))

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + self.pde.spatial_dim + 1 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )
        
        # TODO Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==10): # NEW ADD
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 18, stride=5),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window==25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
            
    def __repr__(self):
        return f'GNN'
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        # process graph data
        u = data.x # (batch*n_nodes, time_window)
        ## Encode and normalize coordinate information
        pos = data.pos # (batch*n_nodes, 1+spatial_dim)
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        pos_x = pos[:, 1:]
        bd_low = []
        bd_up = []
        for low, up in self.pde.spatial_domain:
            bd_low.append(low)
            bd_up.append(up)
        bd_low = torch.tensor(bd_low).to(pos_x.device)
        bd_up = torch.tensor(bd_up).to(pos_x.device)
        pos_x = (pos_x - bd_low) / (bd_up - bd_low)
        edge_index = data.edge_index
        batch = data.batch
        ## Encode equation parameters (modified)
        variables = pos_t # (batch*n_nodes, 1)
        variables = torch.cat((variables, data.variables), -1)

        # Encoder
        # print(u.shape, pos_x.shape, variables.shape)
        # print(u.dtype, pos_x.dtype, variables.dtype)
        node_input = torch.cat((u, pos_x, variables), -1) # variables include t
        h =self.embedding_mlp(node_input)

        # Processor (message passing)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)
        
        # Decoder
        dt = (self.pde.tmax - self.pde.tmin) / self.pde.resolution_t
        dt = (torch.ones(1, self.time_window) * dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        diff = self.output_mlp(h[:, None]).squeeze(1) # study diff (residual): (batch*n_nodes, tw)
        ## the last time step: (batch*n_nodes) -> (tw, batch*n_nodes) -> (batch*n_nodes, tw)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out