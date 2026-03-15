from torchvision import datasets,transforms
from torch.utils.data import Dataset
from s01_preprocess import get_train_transforms,get_test_transforms,get_validation_transforms,compute_stats
from s01_config import Config
from typing import Tuple
from s03_utils import create_train_val_split,assign_transforms_to_subsets,TransformedSubset

cfg=Config()

def downloader(cfg:Config) -> Tuple[Dataset,Dataset,Dataset] :
    """
    Returns the train, validation, test datasets and installs them in path specified
    by cfg.path

    Args:
        cfg(Config):The configuration class that houses all the hyperparameters

    Returns:
        train_dataset, validation, test_dataset(Pytorch Dataset objects)
    """
    # dataset without transforms to capture stats
    full_train_dataset=datasets.GTSRB(
    root=cfg.path,
    split="train",
    transform=transforms.ToTensor(),
    download=True
    )

    # compute stats on the downloaded dataset
    mean,std=compute_stats(full_train_dataset)

    # print the values
    print(f"the dataset mean is {mean}; and the standard deviation is {std}")

    # redownload again without transforms
    full_train_dataset=datasets.GTSRB(
    root=cfg.path,
    split="train",
    transform=None,
    download=True
    )

    # Apply the train-val split in the ratio of 90:10 on the original dataset
    train_dataset,val_dataset=create_train_val_split(
        dataset=full_train_dataset,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    train_set, val_set=assign_transforms_to_subsets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_transform=get_train_transforms(mean,std),
        val_transform=get_validation_transforms(mean,std),
    )


    test_set=datasets.GTSRB(
        root=cfg.path,
        split="test",
        transform=get_test_transforms(mean,std),
        download=True
    )

    return train_set,val_set,test_set
