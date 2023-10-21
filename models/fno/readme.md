train from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_diff-sorp.yaml | tee ./log/train/1D_diff-sorp.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_Adv.yaml | tee ./log/train/1D_Adv.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_Bgs.yaml | tee ./log/train/1D_Burgers.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_SWE.yaml | tee ./log/train/2D_SWE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_diff-react.yaml | tee ./log/train/1D_diff-react.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_CFD.yaml | tee ./log/train/1D_CFD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_diff-react.yaml | tee ./log/train/2D_diff-react.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Darcy.yaml | tee ./log/train/2D_Darcy.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_CFD.yaml | tee ./log/train/2D_CFD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Bgs.yaml | tee ./log/train/2D_Bgs.out(todo)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_Allen-Cahn.yaml | tee ./log/train/Allen-Cahn.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_Cahn-Hilliard.yaml | tee ./log/train/Cahn-Hilliard.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_convec-diff.yaml | tee ./log/train/convec-diff.out
```

test
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_Adv.yaml | tee ./log/test/1D_Adv.out (done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_diff-react.yaml | tee ./log/test/2D_diff-react.out(done) 
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_SWE.yaml | tee ./log/test/2D_SWE.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_CFD.yaml | tee ./log/test/2D_CFD.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_Bgs.yaml | tee ./log/test/1D_Burgers.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_CFD.yaml | tee ./log/test/1D_CFD.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_diff-react.yaml | tee ./log/test/1D_diff-react.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_diff-sorp.yaml | tee ./log/test/1D_diff-sorp.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Darcy.yaml | tee ./log/test/2D_Darcy.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_Allen-Cahn.yaml | tee ./log/test/Allen-Cahn.out(done)
CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_Cahn-Hilliard.yaml | tee ./log/test/Cahn-Hilliard.out(done)
```

