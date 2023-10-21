#1d_burger
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Burgers_1D.yaml | tee log/burger.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Burgers_1D.yaml | tee log/burger_4x.out

#1d_advection
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Adv.yaml | tee log/adv.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Adv.yaml | tee log/adv_4x.out

#1d_diff-sorp
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-sorp_1d.yaml | tee log/diff-sorp_1d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_diff-sorp_1d.yaml | tee log/diff-sorp_1d_4x.out
#1d_diff-react
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-react_1d.yaml | tee log/diff-react_1d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_diff-react_1d.yaml | tee log/diff-react_1d_4x.out

#1d_CFD
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_CFD_1d.yaml | tee log/CFD_1d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_CFD_1d.yaml | tee log/CFD_1d_4x.out
#1d_Cahn_Hilliard
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Cahn-Hilliard.yaml | tee log/CH_1d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Cahn-Hilliard.yaml | tee log/CH_1d_4x.out
#1d_Allen_Cahn
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Allen-Cahn.yaml | tee log/AC_1d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Allen-Cahn.yaml | tee log/AC_1d_4x.out
#2d DarcyFlow
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Darcyflow.yaml | tee log/darcy.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Darcyflow.yaml | tee log/darcy_4x.out

#2d CFD
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_CFD_2d.yaml | tee log/CFD_2d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_CFD_2d_init10.yaml | tee log/CFD_2d_init10.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_CFD_2d.yaml | tee log/CFD_2d_4x.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_CFD_2d_init10.yaml | tee log/CFD_2d_init10_4x.out

#2d shallow water
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_SWE_2d.yaml | tee log/SWE_2d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_SWE_2d.yaml | tee log/SWE_2d_4x.out

#2d_diff-react
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-react_2d.yaml | tee log/diff-react_2d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_diff-react_2d.yaml | tee log/diff-react_2d_4x.out

#2d_burgers
# 2x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Burgers_2D.yaml | tee log/burger_2d.out
# 4x
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_4x/config_Burgers_2D.yaml | tee log/burger_2d_4x.out

