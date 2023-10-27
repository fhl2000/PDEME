# coding=utf-8

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
from scipy.stats import qmc

""" 
PINNDatasetSingle apply to 1DAdvection
PINNDatasetMult apply to 1Ddiff-sorp

1D Advection: 
<KeysViewHDF5 ['t-coordinate', 'tensor', 'x-coordinate']>
<HDF5 dataset "t-coordinate": shape (202,), type "<f4">
<HDF5 dataset "tensor": shape (10000, 201, 1024), type "<f4">
<HDF5 dataset "x-coordinate": shape (1024,), type "<f4">

1D diffusion sorption equation:
<KeysViewHDF5 ['0000', '0001', ..., '9999']
<HDF5 dataset "data": shape (101, 1024, 1), type "<f4">
<HDF5 dataset "x": shape (1024,), type "<f4">
<HDF5 dataset "t": shape (101,), type "<f4">
"""

class PINNDataset1D(Dataset):
    def __init__(self, filename, seed):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """
        self.seed = seed

        # load data file
        root_path = os.path.abspath("/home/zhouziyang/hcy")
        data_path = "./1D_Advection_Sols_beta0.1.hdf5" # os.path.join(root_path, filename)
        with h5py.File(data_path, "r") as h5_file:
            seed_group = h5_file[seed]
            # print(seed_group.attrs["config"])
            # extract config
            self.config = yaml.load(seed_group.attrs["config"], Loader=yaml.SafeLoader)

            # build input data from individual dimensions
            # dim x = [x]
            self.data_grid_x = torch.tensor(seed_group["grid"]["x"], dtype=torch.float)
            # # dim t = [t]
            self.data_grid_t = torch.tensor(seed_group["grid"]["t"], dtype=torch.float)

            XX, TT = torch.meshgrid(
                [self.data_grid_x, self.data_grid_t],
                indexing="ij",
            )

            self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

            self.data_output = torch.tensor(
                np.array(seed_group["data"]), dtype=torch.float
            )

            # permute from [t, x] -> [x, t]
            permute_idx = list(range(1, len(self.data_output.shape) - 1))
            permute_idx.extend(list([0, -1]))
            self.data_output = self.data_output.permute(permute_idx)

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_t))
        test_input_t = self.data_input[:, 1].reshape((n_x, n_t))
        test_output = self.data_output.reshape((n_x, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, -n_last_time_steps:]
        test_input_t = test_input_t[:, -n_last_time_steps:]
        test_output = test_output[:, -n_last_time_steps:, :]

        test_input = torch.vstack([test_input_x.ravel(), test_input_t.ravel()]).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked.T

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        return raveled_tensor.reshape((1, n_x, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(
            self.config["sim"]["x_left"],
            self.config["sim"]["x_right"],
            self.config["sim"]["xdim"],
        )
        # xx, yy = np.meshgrid(x_space, y_space)

        tt = np.ones_like(x_space) * time
        val_input = np.vstack((x_space, tt)).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx].unsqueeze(1)

    def get_name(self):
        return self.config["name"]


class PINNDatasetDiffSorption(PINNDataset1D):
    def __init__(self, filename, seed):
        super().__init__(filename, seed)

        # ravel data
        self.data_output = self.data_output.ravel()

    def get_initial_condition(self):
        # Generate initial condition
        Nx = self.config["sim"]["xdim"]

        np.random.seed(self.config["sim"]["seed"])

        u0 = np.ones(Nx) * np.random.uniform(0, 0.2)

        return (self.data_input[:Nx, :], np.expand_dims(u0, 1))

class PINNDataset1Dpde(Dataset):
    def __init__(self, filename, root_path='./datasets/', val_batch_idx=-1, vol_size=1, vol_ep=1.e-3):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """

        self.vol_size = vol_size   
        self.vol_ep = vol_ep

        # load data file
        data_path = os.path.join(root_path, filename)
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        # dim x = [x]
        self.data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
        self.dx = self.data_grid_x[1] - self.data_grid_x[0]
        self.xL = self.data_grid_x[0] - 0.5 * self.dx
        self.xR = self.data_grid_x[-1] + 0.5 * self.dx
        self.xdim = self.data_grid_x.size(0)
        # # dim t = [t]
        self.data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)

        # main data
        keys = list(h5_file.keys())
        keys.sort()
        print(keys)
        if 'tensor' in keys:
            # print(h5_file["tensor"].shape)
            # print(self.data_grid_t.shape)
            # print(self.xdim)
            self.data_output = torch.tensor(np.array(h5_file["tensor"][val_batch_idx]),
                                            dtype=torch.float)
            # print(self.data_output.shape)
            # print(self.data_output)
            # permute from [t, x] -> [x, t]
            self.data_output = self.data_output.T

            # for init/boundary conditions
            self.init_data = self.data_output[..., 0, None]
            self.bd_data_L = self.data_output[0, :, None]
            self.bd_data_R = self.data_output[-1, :, None]

        else:
            _data1 = np.array(h5_file["density"][val_batch_idx])
            _data2 = np.array(h5_file["Vx"][val_batch_idx])
            _data3 = np.array(h5_file["pressure"][val_batch_idx])
            _data = np.concatenate([_data1[...,None], _data2[...,None], _data3[...,None]], axis=-1)
            # permute from [t, x] -> [x, t]
            _data = np.transpose(_data, (1, 0, 2))

            self.data_output = torch.tensor(_data, dtype=torch.float)
            del(_data, _data1, _data2, _data3)

            # for init/boundary conditions
            self.init_data = self.data_output[:, 0]
            self.bd_data_L = self.data_output[0]
            self.bd_data_R = self.data_output[-1]

        self.tdim = self.data_output.size(1)
        self.data_grid_t = self.data_grid_t[:self.tdim]

        XX, TT = torch.meshgrid(
            [self.data_grid_x, self.data_grid_t],
            indexing="ij",
        )

        self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

        h5_file.close()
        if 'tensor' in keys:
            self.data_output = self.data_output.reshape(-1, 1)
        else:
            self.data_output = self.data_output.reshape(-1, 3)
        

    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)

    def get_boundary_condition(self):
        # return (self.data_grid_t[:self.nt, None], self.bd_data_L, self.bd_data_R)
        return (self.data_input[:self.tdim, :], self.bd_data_L, self.data_input[-1-self.tdim:-1, :], self.bd_data_R)

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_t))
        test_input_t = self.data_input[:, 1].reshape((n_x, n_t))
        test_output = self.data_output.reshape((n_x, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, -n_last_time_steps:]
        test_input_t = test_input_t[:, -n_last_time_steps:]
        test_output = test_output[:, -n_last_time_steps:, :]

        test_input = torch.vstack([test_input_x.ravel(), test_input_t.ravel()]).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        return raveled_tensor.reshape((1, n_x, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(self.xL, self.xR, self.xdim)
        # xx, yy = np.meshgrid(x_space, y_space)

        tt = np.ones_like(x_space) * time
        val_input = np.vstack((x_space, tt)).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx]


class PINNDatasetSingle(Dataset):
    def __init__(self, file_name,
                 saved_folder,
                 initial_step=10,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1):
        
        # file path, HDF5 file is assumed
        file_path = os.path.abspath(saved_folder + file_name)

        # read data from HDF5 file 
        with h5py.File(file_path, 'r') as f:
            ## data dim: [num_sample, t, x1, ..., xd] (The field dimension is 1) 
            # or [num_sample, t, x1, ..., xd, v] (The field dimension is v)
            _data = f['tensor'][...]
            if len(_data.shape) == 3:  # 1D
                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data[:, :, :], (0, 2, 1))
                self.data = _data[:, :, :, None]  # batch, x, t, ch

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

        # time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx,...,:self.initial_step,:], self.data[idx]
    
    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)
    


class PINNDatasetMult(Dataset):
    def __init__(self, file_name, 
                 saved_folder,
                 initial_step=10,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1
                ):
        # file path, HDF5 file is assumed
        self.file_path = os.path.abspath(saved_folder + file_name)
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as f:
            seed_list = sorted(f.keys())
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
            
        # time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.seed_list)
    

    def __getitem__(self, idx):
        # open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.seed_list[idx]]
            ## data dim = [t, x1, ..., xd, v]
            data = seed_group["data"][...]
            data = torch.tensor(data)
            ## convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

        return data[...,:self.initial_step,:], data

def sampleCubeQMC(dim, l_bounds, u_bounds, expon=100):
    '''Quasi-Monte Carlo Sampling

    Get the sampling points by quasi-Monte Carlo Sobol sequences in dim-dimensional space. 

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        expon:    The number of sample points will be 2^expon

    Returns:
        numpy.array: An array of sample points
    '''
    sampler = qmc.Sobol(d=dim, scramble=False)
    sample = sampler.random_base2(expon)
    data = qmc.scale(sample, l_bounds, u_bounds)
    data = np.array(data)
    return data[1:]

class DFVMsolver():
    def __init__(self, dim, device) -> None:
        self.vol_size = 2
        self.vol_ep   = 1.e-4
        self.device = device
        l_bounds = [-1 for _ in range(dim)]
        u_bounds = [ 1 for _ in range(dim)]
        self.Int = sampleCubeQMC(dim, l_bounds, u_bounds, self.vol_size) * self.vol_ep

    def get_vol_data(self, x):
        x_shape = x.shape
        x = x.cpu().numpy()
        x = np.expand_dims(x, axis=1)
        x = np.broadcast_to(x, (x_shape[0], len(self.Int), x_shape[1]))
        x = x + self.Int
        return torch.from_numpy(x).to(self.device).requires_grad_(True)
    
    def get_vol_data2(self, x):
        bound = torch.tensor([self.vol_ep, 0]).to(self.device)
        xL = x - bound
        xR = x + bound 
        return xL.detach().requires_grad_(True), xR.detach().requires_grad_(True)

    def get_len(self):
        return self.vol_ep*2



# test
if __name__ == "__main__":
    dfvm = DFVMsolver(1, "cuda:0")
    print(dfvm.Int)
    print(dfvm.get_len())
    x = torch.tensor([[0.5, 0.5], [1, 1]]).to("cuda:0")
    print(dfvm.get_vol_data2(x))
    # # test PINNDatasetSingle
    # flnm = "1D_Advection_Sols_beta1.0.hdf5"
    # base_path = "/data1/zhouziyang/datasets/pdebench/1D/Advection/Train/"
    # dataset = PINNDatasetSingle(flnm, base_path)

    # # # test PINNDatasetMult
    # # flnm = "1D_diff-sorp_NA_NA.h5"
    # # base_path = "/home/zhouziyang/PDEBench/pdebench/data/1D/diffusion-sorption/"
    # # dataset = PINNDatasetMult(flnm, base_path)

    # # dataloader
    # batch_size = 16
    # num_workers = 4
    # iter_num = 10
    # dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    # print("The number of samples is", len(dataset))
    # for batch, (x, y) in enumerate(dataloader):
    #     if batch > iter_num:
    #         break
    #     x = x.to("cuda:0")
    #     y = y.to("cuda:0")
    #     print(x.shape, y.shape)