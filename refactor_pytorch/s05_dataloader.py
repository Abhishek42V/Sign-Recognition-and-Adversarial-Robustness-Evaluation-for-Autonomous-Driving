import torch
from torch.utils.data import DataLoader,Dataset
from s01_config import Config
from s03_utils import worker_init_fn



def get_train_loader(train_dataset:Dataset,cfg:Config):
    """
    Returns a dataloaderobject given a training dataset
    """
    obj=DataLoader(train_dataset,batch_size=cfg.batch_size_train,
                   shuffle=True,num_workers=cfg.num_workers,worker_init_fn=worker_init_fn,
                   generator=torch.Generator().manual_seed(42),
                   persistent_workers=(cfg.num_workers>0))
    return obj

def get_val_loader(val_dataset:Dataset,cfg:Config):
    """
    Returns a dataloaderobject given a validation dataset
    """
    obj=DataLoader(val_dataset,batch_size=cfg.batch_size_val,
                   shuffle=False,num_workers=cfg.num_workers,worker_init_fn=worker_init_fn,
                   generator=torch.Generator().manual_seed(42),
                   persistent_workers=(cfg.num_workers>0))
    return obj

def get_test_loader(test_dataset:Dataset,cfg:Config):
    """
    Returns a dataloaderobject given a test dataset
    """
    obj=DataLoader(test_dataset,batch_size=cfg.batch_size_test,shuffle=False,
                   num_workers=cfg.num_workers,
                    worker_init_fn=worker_init_fn,
                    )
    return obj

