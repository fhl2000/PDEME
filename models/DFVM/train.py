# coding=utf-8
"""
Three different training methods: 
1. single step training
2. unrolled training
3. pushforward training
"""
import torch
import numpy as np
import yaml
import torch
import torch.nn as nn
from metrics import *
from torch.utils.data import DataLoader
from DFVM import MLP
from utils import PINNDataset1Dpde, PINNDatasetDiffSorption, PINNDatasetMult, DFVMsolver
# import pde_definitions

from pde_definitions import (
    pde_diffusion_reaction,
    pde_swe2d,
    pde_diffusion_sorption,
    pde_swe1d,
    pde_adv1d,
    pde_diffusion_reaction_1d,
    pde_burgers1D,
    pde_CFD1d,
    pde_CFD2d,
    pde_CFD3d,
    pde_Allen_Cahn,
    pde_Cahn_Hilliard
)
from metric import L2RE, MaxError

def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and np.isclose(x[0], xL)) or (on_boundary and np.isclose(x[0], xR))


def setup_pde1D(filename="1D_Advection_Sols_beta4.0.hdf5",
                root_path='./',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=1,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                if_periodic_bc=True,
                aux_params=[0.1]):

    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    if filename[0] == 'R':
        pde = lambda x, model : pde_diffusion_reaction_1d(x, model, aux_params[0], aux_params[1])
    else:
        if filename.split('_')[1]=="Allen-Cahn":
            pde = lambda x, model: pde_Allen_Cahn(x, model, aux_params[0], aux_params[1])
        elif filename.split('_')[1]=="Cahn-Hilliard":
            pde = lambda x, model: pde_Cahn_Hilliard(x, model, aux_params[0], aux_params[1])
        elif filename.split('_')[1][0]=='A':
            pde = lambda x, model: pde_adv1d(x, model, aux_params[0])
        elif filename.split('_')[1][0] == 'B':
            pde = lambda x, y: pde_burgers1D(x, model, aux_params[0])
        elif filename.split('_')[1][0]=='C':
            pde = lambda x, model: pde_CFD1d(x, model, aux_params[0])
        else:
            pde = lambda x, y: pde_diffusion_sorption(x, y)
    if filename.split('_')[1][0]!='D':
        dataset = PINNDataset1Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    IC = {"initial_input": initial_input, "initial_u": initial_u}
    # prepare boundary condition
    if if_periodic_bc:
        if filename.split('_')[1][0] == 'C':
            pass
            # raise NotImplementedError   
    else:
        bd_input_L, bd_uL, bd_input_R, bd_uR = dataset.get_boundary_condition()
        BC = {"bd_input_L": bd_input_L, "bd_uL":bd_uL, "bd_uR": bd_uR}

    # net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    # model = dde.Model(data, net)
    model = MLP(in_channels=input_ch, out_channels=output_ch, hidden_width=hidden_ch)

    return pde, dataset, model


def loss_compute(model, X_inn, X_inL, X_inR, X_init, X_bdL, X_bdR, U_init, U_bdL, U_bdR, pde):
    # Y = model(X_inn)
    loss_res = 0 # torch.mean(pde((X_inn,X_inL,X_inR), model)**2)

    loss_ic = torch.mean((model(X_init) - U_init)**2)

    loss_bcL = torch.mean((model(X_bdL) - U_bdL)**2)
    loss_bcR = torch.mean((model(X_bdR) - U_bdR)**2)

    loss = loss_res + loss_ic + loss_bcL + loss_bcR

    return loss

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def main(args):
    # init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    # model_name = "dfvm"

    # data get dataloader
    set_seed(args["seed"])

    val_num = args["val_num"]
    for val_batch_idx in range(val_num):
        pde, data, model = setup_pde1D(filename=args["filename"],
                                    root_path=args["root_path"],
                                    input_ch=args["input_ch"],
                                    output_ch=args["output_ch"],
                                    if_periodic_bc=args["if_periodic_bc"],
                                    val_batch_idx=val_batch_idx,
                                    aux_params=args["aux_params"],
                                    )
        # pde, data, model = setup_diffusion_sorption(filename=args["filename"],
        #                             seed=args["seed"]
        #                             )
        # nn.init.xavier_normal(model.parameters())
        model = model.to(device)
        DFVM_solver = DFVMsolver(1, device)
        X_inn = data.data_input
        choice = np.random.choice(np.arange(0, X_inn.shape[0]), 1000, replace=False)
        X_inn = X_inn[choice, :]
        X_inL, X_inR = DFVM_solver.get_vol_data2(X_inn)
        X_init, U_init = data.get_initial_condition()
        X_bdL, U_bdL, X_bdR, U_bdR = data.get_boundary_condition()
        X_inn  = X_inn.requires_grad_(True).to(device)
        X_inL  = X_inL.requires_grad_(True).to(device)
        X_inR  = X_inR.requires_grad_(True).to(device)
        X_init = X_init.requires_grad_(True).to(device)
        X_bdL  = X_bdL.requires_grad_(True).to(device)
        X_bdR  = X_bdR.requires_grad_(True).to(device)
        U_bdL  = U_bdL.to(device)
        U_bdR  = U_bdR.to(device)
        U_init = U_init.to(device)

        # set some train args
        Adam_iter = args["epochs"]   # stop when iter > Adam_iter
        optimizerAdam = torch.optim.Adam(
            model.parameters(), 
            lr=args["learning_rate"]
        )

        model.train()
        for step in range(Adam_iter+1):
            # Backward and optimize
            optimizerAdam.zero_grad()
            lossAdam = loss_compute(model, X_inn, X_inL, X_inR, X_init, X_bdL, X_bdR, U_init, U_bdL, U_bdR, pde)
            if step % 100 == 0:
                print('Iter %d, LossAdam: %.5e' % (step, lossAdam.item()))

            lossAdam.backward() # retain_graph = True)
            optimizerAdam.step()

        n_components = args["output_ch"]
        test_input, test_gt = data.get_test_data(
            n_last_time_steps=20, n_components=n_components
        )
        test_pred = model(test_input.to(device))

        # prepare data for metrics eval
        test_pred = data.unravel_tensor(
            test_pred, n_last_time_steps=20, n_components=n_components
        ).detach().cpu()
        test_gt = data.unravel_tensor(
            test_gt, n_last_time_steps=20, n_components=n_components
        ).detach().cpu()

        L2re = L2RE(test_gt, test_pred)
        MaxE = MaxError(test_gt, test_pred)
        print("L2RE: ", L2re)
        print("MaxE: ", MaxE)
    

if __name__ == "__main__":
    with open("config_pinn_pde1d.yaml", 'r') as f:
        args = yaml.safe_load(f)
    print(args)
    main(args)