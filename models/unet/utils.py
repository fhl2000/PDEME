# coding=utf-8

import os
import h5py
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import time

"""
1D Advection: 
<KeysViewHDF5 ['t-coordinate', 'tensor', 'x-coordinate']>
<HDF5 dataset "t-coordinate": shape (202,), type "<f4">
<HDF5 dataset "tensor": shape (10000, 201, 1024), type "<f4">
<HDF5 dataset "x-coordinate": shape (1024,), type "<f4">

1D diffusion sorption:
<KeysViewHDF5 ['0000', '0001', ..., '9999']
<KeysViewHDF5 ['data', 'grid']>
<HDF5 dataset "data": shape (101, 1024, 1), type "<f4">
<HDF5 dataset "x": shape (1024,), type "<f4">
<HDF5 dataset "t": shape (101,), type "<f4">

1D Burgers:
<KeysViewHDF5 ['t-coordinate', 'tensor', 'x-coordinate']>
<HDF5 dataset "t-coordinate": shape (202,), type "<f4">
<HDF5 dataset "tensor": shape (10000, 201, 1024), type "<f4">
<HDF5 dataset "x-coordinate": shape (1024,), type "<f4">

2D SWE
<KeysViewHDF5 ['0000', '0001', ..., '0999']
<KeysViewHDF5 ['data', 'grid']>
<HDF5 dataset "data": shape (101, 128, 128, 1), type "<f4">
<HDF5 dataset "x": shape (128,), type "<f4">
<HDF5 dataset "y": shape (128,), type "<f4">
<HDF5 dataset "t": shape (101,), type "<f4">

2D Diffusion-Reaction
<KeysViewHDF5 ['0000', '0001', ..., '0999']
<KeysViewHDF5 ['data', 'grid']>
<HDF5 dataset "data": shape (101, 128, 128, 2), type "<f4">
<HDF5 dataset "x": shape (128,), type "<f4">
<HDF5 dataset "y": shape (128,), type "<f4">
<HDF5 dataset "t": shape (101,), type "<f4">

1D Compressible NS
<KeysViewHDF5 ['Vx', 'density', 'pressure', 't-coordinate', 'x-coordinate']>
<HDF5 dataset "Vx": shape (10000, 101, 1024), type "<f4">
<HDF5 dataset "density": shape (10000, 101, 1024), type "<f4">
<HDF5 dataset "pressure": shape (10000, 101, 1024), type "<f4">
<HDF5 dataset "t-coordinate": shape (102,), type "<f4">
<HDF5 dataset "x-coordinate": shape (1024,), type "<f4">

2D Compressible NS
<KeysViewHDF5 ['Vx', 'Vy', 'density', 'pressure', 't-coordinate', 'x-coordinate', 'y-coordinate']>
<HDF5 dataset "Vx": shape (10000, 21, 128, 128), type "<f4">
<HDF5 dataset "Vy": shape (10000, 21, 128, 128), type "<f4">
<HDF5 dataset "density": shape (10000, 21, 128, 128), type "<f4">
<HDF5 dataset "pressure": shape (10000, 21, 128, 128), type "<f4">
<HDF5 dataset "t-coordinate": shape (22,), type "<f4">
<HDF5 dataset "x-coordinate": shape (128,), type "<f4">
<HDF5 dataset "y-coordinate": shape (128,), type "<f4">

2D Darcy Flow
<KeysViewHDF5 ['nu', 't-coordinate', 'tensor', 'x-coordinate', 'y-coordinate']>
<HDF5 dataset "nu": shape (10000, 128, 128), type "<f4">
<HDF5 dataset "t-coordinate": shape (10,), type "<f4">
<HDF5 dataset "tensor": shape (10000, 1, 128, 128), type "<f4">
<HDF5 dataset "x-coordinate": shape (128,), type "<f4">
<HDF5 dataset "y-coordinate": shape (128,), type "<f4">

1D Diffusion Reaction
<KeysViewHDF5 ['t-coordinate', 'tensor', 'x-coordinate']>
<HDF5 dataset "t-coordinate": shape (102,), type "<f4">
<HDF5 dataset "tensor": shape (10000, 101, 1024), type "<f4">
<HDF5 dataset "x-coordinate": shape (1024,), type "<f4">

1D Allen Cahn

1D Cahn Hilliard

2D Burgers
<KeysViewHDF5 ['0000', '0001', ..., '0999']
<KeysViewHDF5 ['data', 'grid']>
<HDF5 dataset "data": shape (101, 128, 128, 2), type "<f4">
<HDF5 dataset "x": shape (128,), type "<f4">
<HDF5 dataset "y": shape (128,), type "<f4">
<HDF5 dataset "t": shape (101,), type "<f4">
"""

class UNetDatasetSingle(Dataset):
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
            if "tensor" not in f.keys(): # CFD datasets
                spatial_dim = len(f["density"].shape) - 2
                self.data = None
                if spatial_dim == 1:
                    for i, key in enumerate(["density", "pressure", "Vx"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                        _data = np.transpose(_data, (0, 2, 1))
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(3)
                            self.data = np.zeros(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                elif spatial_dim == 2:
                    for i, key in enumerate(["density", "pressure", "Vx", "Vy"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(4)
                            self.data = np.zeros(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                else: # spatial_dim == 3
                    pass
            else:
                _data = np.array(f["tensor"], dtype=np.float32)
                if len(_data.shape) == 3:  # 1D
                    # data dim: (num_sample, t, x)
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                    ## convert to (num_sample, x, t)
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # (num_sample, x, t, 1)
                elif len(_data.shape) == 4:
                    if "nu" in f.keys(): # 2D darcy flow
                        # label: (num_sample, t, x1, x2)
                        _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
                        ## convert to (num_sample, x1, ..., xd, t)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = _data
                        # nu: (num_sample, x1, x2)
                        _data = np.array(f['nu'], dtype=np.float32)
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution] # (num_sample, t, x1, x2)
                        ## convert to (num_sample, x1, ..., xd, t)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # (num_sample, x1, x2, t, v)
                    # TODO 2D
                    else:
                        pass
                # TODO 3D
                else:
                    pass

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
    


class UNetDatasetMult(Dataset):
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
            # data = seed_group["data"][...]
            data = np.array(seed_group["data"], dtype=np.float32)
            if len(data.shape) == 3: # 1D
                data = data[::self.reduced_resolution_t, ::self.reduced_resolution, :]
            elif len(data.shape) == 4: # 2D
                data = data[::self.reduced_resolution_t, ::self.reduced_resolution, ::self.reduced_resolution, :]
            else: # TODO 3D
                pass
            data = torch.tensor(data)
            ## convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

        return data[..., :self.initial_step, :], data


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


# test
if __name__ == "__main__":
    start_time = time.time()

    # # test UNetDatasetSingle
    # flnm = "1D_Advection_Sols_beta1.0.hdf5"
    # base_path = "/data1/zhouziyang/datasets/pdebench/1D/Advection/Train/"
    # dataset = UNetDatasetSingle(flnm, base_path)

    # test UNetDatasetMult
    flnm = "1D_diff-sorp_NA_NA.h5"
    base_path = "/home/zhouziyang/PDEBench/pdebench/data/1D/diffusion-sorption/"
    dataset = UNetDatasetMult(flnm, base_path, reduced_resolution=4)

    # # test 2 DCFD
    # flnm = "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5"
    # base_path = "/data1/zhouziyang/datasets/pdebench/2D/CFD/2D_Train_Rand/"
    # dataset = UNetDatasetSingle(flnm, base_path)

    # # darcy flow test
    # flnm = "2D_DarcyFlow_beta0.01_Train.hdf5"
    # base_path = "/home/zhouziyang/PDEBench/pdebench/data/2D/DarcyFlow/"
    # dataset = UNetDatasetSingle(flnm, base_path, initial_step=1)

    # dataloader
    batch_size = 16
    num_workers = 0
    iter_num = 10
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    print("The number of samples is", len(dataset))
    for batch, (x, y) in enumerate(dataloader):
        if batch > iter_num:
            break
        x = x.to("cuda:0")
        y = y.to("cuda:0")
        print(x.shape, y.shape)

    end_time = time.time()
    print(end_time - start_time)