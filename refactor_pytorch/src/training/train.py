import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from s05_dataloader import get_train_loader,get_val_loader,get_test_loader
from s09_loss_optimizer import get_optimizer
from s01_config import Config
from s06_model import GTRSB_Model
from s13_scheduler import get_scheduler
from s10_loss_function import get_loss_function

#25.12.09 accuracy addition pending
cfg=Config()
model=MNIST_Model()
optimizer=get_optimizer(model,cfg)
loss_function=get_loss_function()
scheduler=get_scheduler(optimizer=optimizer,cfg=cfg)


def train(train_loader,val_loader,cfg:Config,model,loss_function,optimizer, scheduler,namestring:str):
    #Set model to training mode
    model.train()

    for epoch in range(cfg.epochs):
        total_loss=0.0
        num_batches=0

        for imgs,labels in train_loader:
            imgs,labels=imgs.to(cfg.device),labels.to(cfg.device)

            #zero out all gradients in each epoch
            optimizer.zero_grad()
            predictions=model(imgs)
            loss=loss_function(predictions,labels)

            num_batches+=1

            #compute backward pass
            loss.backward()

            #update the model weights
            optimizer.step()

            #scheduler updates?
            scheduler.step()

            #keep track of the loss
            total_loss+=loss.item()

        avg_loss=total_loss/max(1,num_batches)

        avg_val_loss=evaluate(val_loader=val_loader,cfg=cfg,loss_function=loss_function,namestring="validation",model=model)

        print(f"Epoch {epoch}-> training loss:{avg_loss},validation loss:{avg_val_loss}")

@torch.no_grad()           
def evaluate(val_loader:torch.utils.data.DataLoader,cfg:Config,model:nn.Module,loss_function,namestring:str,print_bool:bool=False):
    """
    Docstring for evaluate
    
    :param val_loader: Description
    :param cfg: Description
    :type cfg: Config
    :param model: Description
    :param loss_function: Description
    """

    #set model to evaluate mode
    model.eval()

    running_loss=0.0
    num_batches=0

    for imgs,labels in val_loader:
        prediction=model(imgs)
        loss=loss_function(prediction, labels)

        running_loss+=loss.item()
        num_batches+=1
    
    avg_loss=running_loss/max(1,num_batches) #ensures fault tolerance

    if print_bool:
        print(f"The {namestring} loss is {avg_loss}")

    return avg_loss
