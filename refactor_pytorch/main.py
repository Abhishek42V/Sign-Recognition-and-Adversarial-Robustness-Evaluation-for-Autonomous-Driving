import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorboard as tensorboard
from config import Config
from data.load_data import downloader
from utils.utils import set_global_seed,worker_init_fn,TransformedSubset,print_dataset_info,logger
from data.dataloader import get_train_loader,get_val_loader,get_test_loader
from models.model import GTRSB_model,model_sanity_checker
from training.train import train,evaluate
from training.evaluate import evaluator
from models.loss_optimizer import get_optimizer,get_loss_function

def main():

    # global seed for reproducibility
    set_global_seed(42)

    # instantiate the config file
    cfg = Config()

    # download the data to the specified path
    trainer,val,test=downloader(cfg)

    # define data loaders
    train_loader = get_train_loader(trainer,cfg)
    val_loader = get_val_loader(val,cfg)
    test_loader = get_test_loader(test,cfg)

    # define the model
    model=GTRSB_model(cfg).to(cfg.device)

    #check sanity of the model
    model_sanity_checker(cfg)

    # define the loss function
    loss_fn=get_loss_function()

    # define optimizer
    optimizer=get_optimizer(model,cfg)

    # training loop
    train(train_loader, val_loader,loss_fn,optimizer,model, cfg, namestring)

    #eval loop on test dataset
    evaluator(test_loader,loss_fn,model,cfg)

    #visualize
    call the helper function

    #save the model
    save it as original_trained_gtrsb
    #generate adversarial samples on the test set?
    test_adversarial_loader=get_adversarial_loader(test_loader)
    #evaluate again
    test(test_adversarial_loader,loss_fn,model,cfg)


    # build adversarial dataset
    #50% train, 50% adversarial 1st iteration , no weight
    #2nd iteration weighted probarbilites
    # 3rd iteration random sampling

    # retrain on the adversarial dataset
    train(adversarial_train_loader,adversarial_val_loader,model,cfg, loss_fn,optimizer)

    # resave as a new model

    # check on the adversarial examples, and the original test dataset
    test(test_adversarial_loader,loss_fn,model,cfg)

    #use tensorboard to compare the 3 models on metrics acrorrs the training,validation,
    #adversarial training loosps

if __name__ == "__main__":
    main()