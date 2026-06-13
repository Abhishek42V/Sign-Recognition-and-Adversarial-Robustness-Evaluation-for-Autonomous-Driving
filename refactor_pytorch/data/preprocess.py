import torch
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from typing import Tuple


def compute_stats(dataset: Dataset)-> Tuple[Tensor,Tensor]:
    """
    Compute channel wise mean and standard deviation for an image dataset
    Args:
        dataset(Dataset): A Dataset onject with images of shape (channel, height, width)
    Returns:
        mean(Tensor):of shape (channel,)
        std(Tensor):of shape (channel,)
    """

    loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=4)

    channel_sum=torch.zeros(3,dtype=torch.float64)
    channel_squared_sum=torch.zeros(3,dtype=torch.float64)
    total_pixels=0

    for images,_ in loader:
        # ensure float
        images=images.float()

        batch_size, channels, height,width=images.shape
        # Creating a flattened view of the image space; shape=(batch,size,height*width)
        images=images.view(batch_size,channels,-1) 

        # Sum over batch and pixel(H*W) dimensions
        channel_sum+=images.sum(dim=(0,2))

        # Sum over squared elements for variance computation
        channel_squared_sum+=(images**2).sum(dim=(0,2))

        # total pixels processed per channel
        total_pixels+=batch_size*height*width
        
    # E[X]
    mean=channel_sum/total_pixels

    # E[X**2]
    squared=channel_squared_sum/total_pixels

    std=torch.sqrt(squared-(mean**2))

    return mean, std
 
def get_train_transforms(mean:Tensor,std:Tensor)->transforms.Compose:
    """Applies the augmentation to the training set given mean,std of input images"""
    
    preprocess=transforms.Compose([
        transforms.RandomAffine(degrees=20, # rotation range +-20 deg
                                translate=(0.15,0.15), # width shift, height shift, +-15%
                                scale=(0.85,1.15), #zoom in/out 15%
                                shear=15,  
                                fill=0), # fill nearest with 0
        transforms.ToTensor(),
        transforms.Normalize((mean.tolist()),(std.tolist()))
    ])

    return preprocess

def get_validation_transforms(mean:Tensor,std:Tensor)->transforms.Compose:
    """Applies the transformation to the validation set given mean,std of input images"""
    preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean.tolist()),(std.tolist()))
    ])

    return preprocess

def get_test_transforms(mean:Tensor,std:Tensor)->transforms.Compose:
    """Applies the transformation to the test set given mean,std of input images"""
    preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean.tolist()),(std.tolist()))
    ])

    return preprocess