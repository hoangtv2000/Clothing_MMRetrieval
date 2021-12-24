import torch
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

def seeding(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    SEED = config.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)



def get_checkpoints():
    """List all of checkpoints.
    """
    list = glob.glob(f'../RETRIEVAL/checkpoints/*')
    for i, x in enumerate(list):
        [print(i+1 ,':    ' ,x.split('\\')[-1])]



def optimizer_to_cuda(optimizer, device):
    """Moving optimizer to GPU after loading
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)



def show_demo_test_retrieval(inp, src_string=None, mod_string=None, show=False):
    """Convert augmentated image to raw image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if 'torch' in str(type(inp)):
        inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    if show == False:
        return inp

    else:
        if src_string or mod_string:
            plt.title(str(f'Source: {src_string} \n Mods: {mod_string}'))
        plt.imshow(inp)
        plt.pause(0.001)
