import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import json,csv
from torch.utils.data import Dataset,random_split,Subset
from refactor_pytorch.config.config import Config
from pathlib import Path
from typing import Dict, List, Any,Optional,Tuple,Sized

cfg=Config()

def create_train_val_split(dataset: Dataset, val_ratio: float=0.2, seed:int=42)->Tuple[Subset,Subset]:
    """
    Splits the Pytorch Dataset into train, validation datasets using the val_ratio
    Args:
        dataset(Dataset): The input pytorch dataset object
        val_ratio(float): The ratio of train:val split is (1-val_ratio):(val_ratio)
        seed(int): seed to ensure reproducibility , defaults to 42
    Returns:
        Tuple(Subset,Subset):Tuple of Train, validation subsets
    """
    if val_ratio<0 or val_ratio>1:
        raise ValueError("val_ratio must be between 0,1")
    
    val_size=int(len(dataset)*val_ratio)  
    train_size=len(dataset)-val_size

    gen = torch.Generator().manual_seed(seed)

    train, val= random_split(dataset,[train_size,val_size],generator=gen)

    return train,val

def set_global_seed(seed:int):
    #setting the seed to ensure reproducibility

    random.seed(seed)  # for any python based randomization
    np.random.seed(seed)  # for numpy based randomization
    torch.manual_seed(seed)  # for randomization on cpu using torch
    torch.cuda.manual_seed_all(seed)  # for randomization on gpu using torch

class TransformedSubset(Subset):
    """
    Custom class that inherits from the original Subset class
    and allows for individual transformations to be applied
    at the time of retrieval
    """
    def __init__(self,subset:Subset, transform):
        super().__init__(subset.dataset,subset.indices)
        self.transform=transform

    def __getitem__(self, idx):
        #delegate to underlying dataset;idx is index within subset
        dataset_idx=self.indices[idx]
        img,label=self.dataset[dataset_idx]
        if self.transform is not None:
            img=self.transform(img)
        return img,label


def assign_transforms_to_subsets(train_dataset:Subset,
        val_dataset:Subset,
        train_transform,
        val_transform)->Tuple[TransformedSubset,TransformedSubset]:
    """
    Assigns transformations to train,validation splits
    of the original training dataset
    """
    train_mod=TransformedSubset(train_dataset,train_transform)
    val_mod=TransformedSubset(val_dataset,val_transform)

    return train_mod,val_mod
 
def worker_init_fn(worker_id:int):
    #defining randomization for each of the workers during I/O to the model
    #Called on each DataLoader worker process spawn.
    #Sets deterministic seeds per-worker to make augmentations reproducible.

    seed=cfg.seed+worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
