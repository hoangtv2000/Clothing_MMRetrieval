import sys, os
import time
import numpy as np
from logger.logger import Logger
from tqdm import tqdm as tqdm
from datetime import datetime

import torch
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torchvision

from utils.util import get_checkpoints, optimizer_to_cuda
from .modelloader import load_checkpoint
from .trainer import train_loop
from dataloader.fashion200k_loader import Fashion200k
from img_text_composition_model.compose_transformers import ComposeTransformers


def create_scheduler_optimizer(model, config):
    """Build optimizer and scheduler.
    """
    opt = config['optimizer']['type']
    lr = config['optimizer']['lr']
    dec = config['optimizer']['weight_decay']

    param_dicts = [{
        'params': [p for p in model.img_model.classifier.parameters()],
        'lr': lr
    }, {'params': [p for p in model.img_model.parameters()],
        'lr': 0.5 * lr
    }, {'params': [p for p in model.parameters()],
        'lr': lr
    }]
    for _, p1 in enumerate(param_dicts):  # remove duplicated params
        for _, p2 in enumerate(param_dicts):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    if (opt == 'AdamW'):
        print("Create optimizer Adam with lr: ", lr)
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=dec)
    else:
        raise NotImplementedError('Only supports AdamW!')
    scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler']['scheduler_factor'],
                                      patience=config['scheduler']['scheduler_patience'],
                                      min_lr=config['scheduler']['scheduler_min_lr'],
                                      verbose=config['scheduler']['scheduler_verbose'])
    return optimizer, scheduler



def load_dataset(config):
    trainset = Fashion200k(config=config, mode='train')
    testset = Fashion200k(config=config, mode='test')

    return trainset, testset



def engine(config):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")

    # Device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.cuda) else "cpu")

    # Dataset
    trainset, testset = load_dataset(config)

    # Model
    model = ComposeTransformers(config=config) #Get all captions

     # Add to GPU (if able)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer, scheduler = create_scheduler_optimizer(model, config['model'])

    # Freeze layers
    for p in model.text_model.parameters():
        p.requires_grad = False

    # Load model from checkpoint if config load = True
    if config.load_for_training:
        print('----- LOADING CHECKPOINTS -----')
        get_checkpoints()
        checkpoint_name = input("Choose one of these checkpoints: ")
        cpkt_fol_name = os.path.join(config.cwd, f'checkpoints/{checkpoint_name}')
        checkpoint_dirmodel = f'{cpkt_fol_name}/latest_checkpoint.pth'
        model, optimizer, _, start_epoch = load_checkpoint(config, checkpoint_dirmodel, model, optimizer, scheduler)
        optimizer_to_cuda(optimizer, device)

        print(f'Optimizer: {optimizer}')
        print(f'Start epoch: {start_epoch}')

    else:
        print('------- TRAINING NEW -------')
        start_epoch = 1
        print(f'Optimizer: {optimizer}')


    model.img_model.freeze_layers()
    start_epoch = 2

    # Create a new checkpoint
    cpkt_fol_name = os.path.join(config.cwd, f'checkpoints/date_{dt_string}')

    log = Logger(path=cpkt_fol_name, name='ComposeTransformers').get_logger()
    log.info(f"Checkpoint Folder {cpkt_fol_name}")

    log.info(f"date and time = {dt_string}")
    log.info(f'pyTorch VERSION: {torch.__version__}')
    log.info(f'CUDA VERSION: {torch.version.cuda}')
    log.info(f'CUDNN VERSION: {torch.backends.cudnn.version()}')
    log.info(f'Number CUDA Devices: {torch.cuda.device_count()}')
    log.info(f'device: {device}')
    log.info(f'all parameters: {sum(p.numel() for p in model.parameters())}')
    log.info(f'trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


    train_loop(config=config, log=log, loss_weights=config.loss_weights, trainset=trainset, testset=testset, \
            model=model, optimizer=optimizer, scheduler=scheduler, cpkt_fol_name=cpkt_fol_name, start_epoch=start_epoch)
