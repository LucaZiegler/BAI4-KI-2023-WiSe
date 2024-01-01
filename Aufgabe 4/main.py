import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Servus(nn.Module):
    def __init__(self):
        super().__init__()

model = Servus()