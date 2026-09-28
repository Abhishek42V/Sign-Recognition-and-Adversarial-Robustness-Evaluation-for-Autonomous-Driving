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

def print_dataset_info(train_dataset, test_dataset):
    """
    Prints information about train and test datasets.
    Accepts Subset, TransformedSubset, or Dataset objects.
    """
    def _info(ds):
        n = len(ds)
        # try to get shape if underlying dataset has .data (MNIST)
        if hasattr(ds, 'data'):
            # full dataset (MNIST) case
            shape = getattr(ds, 'data').shape
        elif isinstance(ds, Subset) and hasattr(ds.dataset, 'data'):
            # Subset wrapping MNIST dataset
            # show original data shape and how many subset elements
            shape = getattr(ds.dataset, 'data').shape
        else:
            shape = None
        return n, shape

    train_n, train_shape = _info(train_dataset)
    test_n, test_shape = _info(test_dataset)

    print("=" * 75)
    print(f"Training Samples: {train_n}")
    if train_shape is not None:
        print(f"Original dataset shape (full): {train_shape}")
    print("=" * 75)
    print(f"Testing Samples: {test_n}")
    if test_shape is not None:
        print(f"Original dataset shape (full): {test_shape}")


# -------------------------
# History logging + plotting (new)
# -------------------------
def init_history() -> Dict[str, List[float]]:
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }


def append_epoch_history(history: Dict[str, List[float]],
                         epoch: int,
                         train_loss: float,
                         val_loss: float,
                         train_acc: float,
                         val_acc: float,
                         lr: float):
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(lr)


def save_history_json(history: Dict[str, List[float]], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def save_history_csv(history: Dict[str, List[float]], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = list(zip(*(history[k] for k in keys)))
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_metrics(history: Dict[str, List[float]],
                 out_path: Optional[str] = None,
                 plot_every: int = 1,
                 title_suffix: str = ""):
    """
    Plot train/val loss, train/val acc, and LR. plot_every controls sampling epochs shown.
    If out_path provided, saves PNG. Otherwise shows plt.show().
    """
    epochs = np.array(history["epoch"])
    if len(epochs) == 0:
        print("plot_metrics: empty history")
        return

    sel = np.arange(0, len(epochs), plot_every)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), tight_layout=True)

    axes[0].plot(epochs[sel], np.array(history["train_loss"])[sel], label="train_loss")
    axes[0].plot(epochs[sel], np.array(history["val_loss"])[sel], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs[sel], np.array(history["train_acc"])[sel], label="train_acc")
    axes[1].plot(epochs[sel], np.array(history["val_acc"])[sel], label="val_acc")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs[sel], np.array(history["lr"])[sel], label="learning_rate")
    axes[2].set_ylabel("LR")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"Training Metrics {title_suffix}".strip())
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


# -------------------------
# Small utility: get lr
# -------------------------
def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        return float(g.get('lr', 0.0))
    return 0.0
