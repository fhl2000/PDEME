#1d_burger
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Burgers_1D.yaml | tee log/burger.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_Burgers_1D.yaml | tee log/burger_ar.out
#1d_advection
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Adv.yaml | tee log/adv.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_Adv.yaml | tee log/adv_ar.out
#1d_diff-sorp
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-sorp_1d.yaml | tee log/diff-sorp_1d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_diff-sorp_1d.yaml | tee log/diff-sorp_1d_ar.out
#1d_diff-react
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-react_1d.yaml | tee log/diff-react_1d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_diff-react_1d.yaml | tee log/diff-react_1d_ar.out
#1d_CFD
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_CFD_1d.yaml | tee log/CFD_1d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_CFD_1d.yaml | tee log/CFD_1d_ar.out
#1d_Cahn_Hilliard
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Cahn-Hilliard.yaml | tee log/CH_1d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_Cahn-Hilliard.yaml | tee log/CH_1d_ar.out
#1d_Allen_Cahn
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Allen-Cahn.yaml | tee log/AC_1d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_Allen-Cahn.yaml | tee log/AC_1d_ar.out
#2d DarcyFlow
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Darcyflow.yaml | tee log/darcy.out

#2d CFD
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_CFD_2d.yaml | tee log/CFD_2d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_CFD_2d.yaml | tee log/CFD_2d_ar.out
#2d shallow water
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_SWE_2d.yaml | tee log/SWE_2d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_SWE_2d.yaml | tee log/SWE_2d_ar.out
#2d_diff-react
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_diff-react_2d.yaml | tee log/diff-react_2d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_diff-react_2d.yaml | tee log/diff-react_2d_ar.out
#2d_burgers
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs/config_Burgers_2D.yaml | tee log/burger_2d.out
CUDA_VISIBLE_DEVICES=0 python3 train.py ./configs_ar/config_Burgers_2D.yaml | tee log/burger_2d_ar.out