import torch, os
import torch.nn as nn
from torch import nn
from functools import partial
from torchvision import models

class ViT(nn.Module):

    def __init__(self, config):
        super(ViT, self).__init__()
        self.config = config

        self.vit = torch.hub.load('facebookresearch/deit:main',
                           'deit_small_patch16_224', pretrained=True)
        self.vit.head = nn.GELU()
        self.classifier = nn.Sequential(nn.Linear(384, 384), nn.Dropout(p=0.25))


    def freeze_layers(self):
        for i in range(self.config.freeze_block):
            if i == self.config.freeze_block-1:
                for p in self.vit.blocks[i].attn.parameters():
                    p.requires_grad = False
            else:
                for p in self.vit.blocks[i].parameters():
                    p.requires_grad = False


    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        return x
