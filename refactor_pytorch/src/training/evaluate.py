import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from s01_config import Config

cfg=Config()

def evaluator(test_loader,model):
    #set model to evaluation mode;
    model.eval()


    correct=0
    total=0
    #set it such that overhead in storage is minimized
    #avoids unnecessary backprop calculations
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs,labels=imgs.to(cfg.device),labels.to(cfg.device)

            predictions=model(imgs)
            _,preds=torch.max(predictions,1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    
    print(f"Accuracy: {100*correct/total: .3f}%")