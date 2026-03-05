import torch
import torch.nn as nn
from s01_config import Config
from torchinfo import summary


class GTRSB_model(nn.Module):
    """
    Custom class definition for the model architecture
    """

    def __init__(self,cfg:Config):
        super(GTRSB_model,self).__init__()

        self.layers=nn.Sequential(
            Conv_block(cfg),
            Conv_block(cfg,same=True),
            Conv_block(cfg,same=True),
            nn.Flatten(),
            FCC_block(cfg),
        )

    def forward(self,x):
        
        assert x.ndim==4,"Input must be (batch,channels,height,width,)"
        assert x.shape==3,"Expected 3 channels corresponding to R,G,B colors"

        x = self.layers(x)

        return x


class Conv_block(nn.Module):
    """
    custom convolution+ max_pooling blocks
    Args:
        cfg:Configuration object that houses the hyperparameters
        same:bool indicating choice of input, output channels for
            different conv block structures
    """
    def __init__(self,cfg:Config,*,same:bool=False):

        super(Conv_block,self).__init__()

        # __init__ local variables
        if not same:
            in1=cfg.channels
            out1=cfg.kernel1
            in2=cfg.kernel1
            out2=cfg.kernel2
        
        else:
            in1=cfg.kernel2
            out1=cfg.kernel2
            in2=cfg.kernel2
            out2=cfg.kernel2

        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in1,out_channels=out1,
                      kernel_size=cfg.kernel_size,stride=cfg.stride_size,
                      padding=cfg.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in2,out_channels=out2,
                      kernel_size=cfg.kernel_size,stride=cfg.stride_size,
                      padding=cfg.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=cfg.kernel_size),
        )

    
    def forward(self,x):
        x = self.block(x)
        return x


class FCC_block(nn.Module):
    """
    custom fully connected layer
    Args
    """
    def __init__(self,cfg:Config):

        super(FCC_block,self).__init__()


        self.block=nn.Sequential(
            nn.LazyLinear(out_features=cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_dim,out_features=cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features=cfg.hidden_dim,out_features=cfg.num_categories),
        )
    
    def forward(self,x):
        x = self.block(x)
        return x


def model_sanity_checker(cfg:Config):
    model = GTRSB_model(cfg)
    summary(model,input_size=(1,3,32,32))