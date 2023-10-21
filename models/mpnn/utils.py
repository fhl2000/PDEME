import h5py
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from typing import Tuple
import random


class PDE(object):
    def __init__(self, 
                 name: str,
                 variables: dict,
                 temporal_domain: Tuple,
                 resolution_t: int,
                 spatial_domain: list,
                 resolution: list,
                 reduced_resolution_t: int=1,
                 reduced_resolution: int=1):
        super().__init__()
        self.name = name
        self.tmin = temporal_domain[0]
        self.tmax = temporal_domain[1]
        self.resolution_t = int(resolution_t / reduced_resolution_t)
        self.spatial_domain = spatial_domain
        self.resolution = [int(res / reduced_resolution)  for res in resolution]
        self.spatial_dim = len(spatial_domain)
        self.variables = variables

    def __repr__(self):
        return self.name
    

class MPNNDatasetSingle(Dataset):
    def __init__(self, 
                 file_name: str,
                 saved_folder: str,
                 reduced_resolution: int=1,
                 reduced_resolution_t: int=1,
                 reduced_batch: int=1,
                 if_test: bool=False,
                 test_ratio: float=0.1,
                 num_samples_max: int=-1,
                 variables: dict={}) -> None:

        super().__init__()

        # file path
        file_path = os.path.abspath(saved_folder + file_name)

        # read data and coordinates from HDF5 file
        with h5py.File(file_path, 'r') as f:
            _data = np.array(f["tensor"], dtype=np.float32)
            if len(_data.shape) == 3:  # 1D
                # coordinates: (x, 1)
                self.coordinates = f["x-coordinate"][::reduced_resolution][:, None]
                # data: (num_sample, t, x)
                self.data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
            elif len(_data.shape) == 4:
                if "nu" in f.keys(): # 2D darcy flow
                    # coordinates: (x1*x2, 2)
                    x = f["x-coordinate"][::reduced_resolution]
                    y = f["y-coordinate"][::reduced_resolution]
                    X, Y = np.meshgrid(x, y)
                    X = X.ravel()
                    Y = Y.ravel() 
                    self.coordinates = np.concatenate((X[..., None], Y[..., None]), axis=-1)
                    # label: (num_sample, t, x1, x2)
                    _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
                    self.data = _data
                    # nu: (num_sample, x1, x2)
                    _data = np.array(f['nu'], dtype=np.float32)
                    _data = _data[::reduced_batch, None, ::reduced_resolution, ::reduced_resolution] # (num_sample, t, x1, x2)
                    # concate and reshape (num_sample, t, x1*x2*...*xd)
                    self.data = np.concatenate([_data, self.data], axis=1)
                    self.data =self.data.reshape((self.data.shape[0], self.data.shape[1], -1))
                # TODO 2D
                else:
                    pass
            # TODO 3D
            else:
                pass

        self.variables = variables

        # Define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        # Construct train/test dataset
        test_idx = int(num_samples_max * (1-test_ratio))
        if if_test:
            self.data = self.data[test_idx:num_samples_max]
        else:
            self.data = self.data[:test_idx]

        # To tensor
        self.data = torch.tensor(self.data)
        self.coordinates = torch.tensor(self.coordinates)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # data: (bs, tw, num_points) coordinates: (bs, num_points, spatial_dim)
        return self.data[idx], self.coordinates, self.variables
    


class MPNNDatasetMult(Dataset):
    def __init__(self, 
                 file_name: str, 
                 saved_folder: str,
                 reduced_resolution: int=1,
                 reduced_resolution_t: int=1,
                 reduced_batch: int=1,
                 if_test: bool=False,
                 test_ratio: float=0.1,
                 num_samples_max: int=-1,
                 variables: dict={}
                ):
        # file path, HDF5 file is assumed
        file_path = os.path.abspath(saved_folder + file_name)
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.variables = variables

        # Extract list of seeds
        self.file_handle = h5py.File(file_path, 'r')
        seed_list = sorted(self.file_handle.keys())
        seed_list = seed_list[::reduced_batch]

        # Define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, len(seed_list))
        else:
            num_samples_max = len(seed_list)

        # Construct test dataset
        test_idx = int(num_samples_max * (1-test_ratio))
        if if_test:
            self.seed_list = np.array(seed_list[test_idx:num_samples_max])
        else:
            self.seed_list = np.array(seed_list[:test_idx])

    def __len__(self):
        return len(self.seed_list)
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        seed_group = self.file_handle[self.seed_list[idx]]
        ## data dim = [t, x1, ..., xd, v] 
        # data = seed_group["data"][...]
        data = np.array(seed_group["data"], dtype=np.float32)
        if len(data.shape) == 3: # 1D
            coordinates = seed_group["x"][::self.reduced_resolution][:, None]
            data = data[::self.reduced_resolution_t, ::self.reduced_resolution, :]
        elif len(data.shape) == 4: # 2D
            x = seed_group["x"][::self.reduced_resolution]
            y = seed_group["y"][::self.reduced_resolution]
            X, Y = np.meshgrid(x, y)
            X = X.ravel()
            Y = Y.ravel()
            coordinates = np.concatenate((X[..., None], Y[..., None]), axis=-1)
            data = data[::self.reduced_resolution_t, ::self.reduced_resolution, ::self.reduced_resolution, :]
        else: # TODO 3D
            pass
        data = torch.tensor(data).squeeze(-1) # default: v=1, (t, x1, ..., xd)
        data = data.reshape((data.shape[0], data.shape[1], -1))
        coordinates = torch.tensor(coordinates)
        return data, coordinates, self.variables
    


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 25) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE to solve
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.nt = pde.resolution_t
        # the number of points per sample
        nx = 1
        for item in self.pde.resolution:
            nx = nx * item
        self.nx = nx
        
        print("nt:", self.nt, "nx:", self.nx)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time steps
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels
    
    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # print("nt:", self.nt, "nx:", self.nx)
        t = torch.linspace(self.pde.tmin, self.pde.tmax, self.nt)
        
        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), ) # u: (bs*nx, tw)
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), ) # u: (bs*nx. tw)
            x_pos = torch.cat((x_pos, x[0]), dim=0)
            t_pos = torch.cat((t_pos, torch.ones(self.nx) * t[step]), dim=0)
            batch = torch.cat((batch, torch.ones(self.nx) * b), dim=0)
        # print("u:", u.shape)
        # print("y:", y.shape)
        # print("x_pos:", x_pos.shape)
        # print("t_pos:", t_pos.shape)
        # print("batch:", batch.shape)
        
        # Calculate the edge_index
        dx = x[0][1][0] - x[0][0][0]
        # print("dx", dx)
        radius = self.n * dx + dx / 10 # modified: ensure to include specified number of neighbors
        edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        # print("edge_index:", edge_index.shape)

        graph = Data(x=u, edge_index=edge_index) # x: (num_nodes, num_node_features) (bs*nx, tw)
        graph.y = y # node-level ground-truth labels
        # Node position matrix with shape [num_nodes, num_dimensions]
        graph.pos = torch.cat((t_pos[:, None], x_pos), 1) # graph.pos: (bs*nx, 1+spatial_dim)
        graph.batch = batch.long()

        # modified
        graph_variables = torch.Tensor()
        for k in variables.keys():
            variable = torch.Tensor()
            for i in batch.long():
                variable = torch.cat((variable, torch.tensor([variables[k][i]], dtype=torch.float32)))
            graph_variables = torch.cat((graph_variables, variable[:, None]), dim=-1)
        # print("graph variable shape:", graph_variables.shape) # (num_nodes, num_variables)
        graph.variables = graph_variables

        return graph
    
    def create_next_graph(self,
                            graph: Data,
                            pred: torch.Tensor,
                            labels: torch.Tensor,
                            steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # output is new input
        graph.x = pred

        t = torch.linspace(self.pde.tmin, self.pde.tmax, self.nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(self.nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph
    

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def to_PDEBench_format(graph_data: torch.Tensor, batch_size: int, pde: PDE):
    """Convert graph data to PDEBench formart

    graph_data (torch.Tensor): input/output data of model with shape (num_points, t)
    batch_size (int): batch size
    pde (PDE): PDE to solve
    """
    assert (len(graph_data.shape) == 2)
    output_shape = [batch_size]
    for v in pde.resolution:
        output_shape.append(v)
    output_shape.append(graph_data.shape[-1])
    return graph_data.reshape(output_shape).unsqueeze(-1)


# test
if __name__ == "__main__":
    file_name = "2D_DarcyFlow_beta1.0_Train.hdf5"
    saved_folder = "/data1/zhouziyang/datasets/pdebench/2D/DarcyFlow/"
    variables = {"beta": 0.1}

    dataset = MPNNDatasetSingle(file_name, saved_folder, variables=variables)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    u, x, variables = next(iter(dataloader))
    print(u.shape, x.shape, variables)