import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training and save checkpoints if validation loss doesn't improve after a given patience.
    """
    def __init__(self, config, logger, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = config.epoch_patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.total_loss_min = np.Inf
        self.delta = delta
        self.logger = logger



    def __call__(self, cpkt_dir, model, optimizer, scheduler, epoch, total_loss):
        params =  {  'cpkt_dir': cpkt_dir,
                     'model': model,
                     'optimizer': optimizer,
                     'scheduler': scheduler,
                     'epoch': epoch,
                     'total_loss': total_loss
                    }
        score = -total_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(**params)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(**params)
            self.counter = 0



    def save_checkpoint(self, cpkt_dir, model, optimizer, scheduler, epoch, total_loss):
        print(f'Validation loss decreased ({self.total_loss_min:.6f} --> {total_loss:.6f}).  Saving model ...')
        save_path = cpkt_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}
        name = os.path.join(cpkt_dir, 'latest_checkpoint.pth')
        print(name)

        torch.save(state, name)
        self.total_loss_min = total_loss
