import numpy as np
import os
import torch
from torch import nn
from torchvision import models



def load_checkpoint(config, checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    """
    checkpoint_loader = torch.load(checkpoint, map_location='cpu')
    print('Checkpoint dist contains: ',checkpoint_loader.keys())

    my_state_dict = model.state_dict()
    for k, v in checkpoint_loader['state_dict'].items():
        if k not in my_state_dict.keys():
            continue
        my_state_dict.update({k: v})

    model.load_state_dict(my_state_dict, strict=True)

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if optimizer != None:
        optimizer.load_state_dict(checkpoint_loader['optimizer'])

    if scheduler != None:
        scheduler.load_state_dict(checkpoint_loader['scheduler'])

    epoch = checkpoint_loader['epoch']

    return model, optimizer, scheduler, epoch
