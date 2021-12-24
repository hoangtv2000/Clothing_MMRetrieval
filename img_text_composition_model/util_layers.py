import torch

class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer.
    """
    def __init__(self):
        super(NormalizationLayer, self).__init__()
        self.norm_scale = torch.nn.Parameter(torch.FloatTensor((1.0,)))


    def forward(self, x):
        self.norm = torch.norm(x, dim=1, keepdim=True).expand_as(x)
        features = self.norm_scale * x / self.norm
        return features
