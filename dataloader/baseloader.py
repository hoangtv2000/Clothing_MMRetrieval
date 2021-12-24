import torch

class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.imgs = []
        self.test_queries = []

    def get_loader(self):
        train_params = {'batch_size': self.config.dataloader.train.batch_size,
                        'shuffle': self.config.dataloader.train.shuffle,
                        'drop_last': True,
                        'num_workers': self.config.dataloader.train.num_workers,
                        'pin_memory': True
        }

        return torch.utils.data.DataLoader(self, **train_params)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError
