import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def compute_stats(dataset):
    """computes mean, std for input dataset"""

    loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=4)

    mean=0.0
    std=0.0
    total_images=0

    for images,_ in loader:

        batch_size=images.size(0) # 512
        channels=images.size(1) # 3
        images=images.view(batch_size,channels,-1) # shape=(512,3,1024*) 1024=32*32

        # compute mean, std over flattened pixel space, and sum over batch size
        mean+=torch.sum(torch.mean(images,dim=2),dim=0)
        std+=torch.sum(torch.std(images,dim=2),dim=0)

        #keep track of total count
        total_images+=batch_size
    
    mean=mean/total_images
    std=std/total_images

    return mean,std


def get_train_transforms(mean,std):
    #augmentation for the training dataset 
    preprocess=transforms.Compose([
        transforms.RandomAffine(degrees=20, # rotation range +-20 deg
                                translate=(0.15,0.15), # width shift, height shift, +-15%
                                scale=(0.85,1.15), #zoom in/out 15%
                                shear=15,  
                                fill=0), # fill nearest with 0
        #transforms.RandomHorizontalFlip(), # randomly flips an image
        transforms.ToTensor(),
        transforms.Normalize((mean),(std))
    ])

    return preprocess

def get_validation_transforms(mean,std):
    preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean),(std))
    ])

    return preprocess

def get_test_transforms(mean,std):
    preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean),(std))
    ])

    return preprocess