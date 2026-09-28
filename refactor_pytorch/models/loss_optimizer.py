import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import GTRSB_model


def get_optimizer(model:GTRSB_model,cfg:Config):
    return optim.Adam(model.parameters(),lr=cfg.lr)

def get_loss_function():
    return nn.CrossEntropyLoss()