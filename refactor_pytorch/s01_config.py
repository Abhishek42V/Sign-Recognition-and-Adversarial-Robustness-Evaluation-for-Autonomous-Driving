import torch

class Config:
    """
    This class houses all the hyperparameters
    """
    def __init__(self):
        # ------pre-modelling------------
        self.path="./gtrsb"
        self.num_workers=4 # Setting up workers that can handle I/O in parallel
        self.val_ratio=0.1 # train,validation split
        self.seed=42 # for ensuring reproducibility

        # --------modelling phase----------
        self.channels=3
        self.kernel1=32 # Number of convolution kernels
        self.kernel2=64 
        self.kernel_size=2
        self.stride_size=1
        self.padding="same"
        self.hidden_dim=256
        self.dropout=0.3
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_categories=43

        # --------training phase-----------
        self.lr=0.001
        self.epochs=15
        self.batch_size_train=32
        self.batch_size_val=32
        self.batch_size_test=32
 
