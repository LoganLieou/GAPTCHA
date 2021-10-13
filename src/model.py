import torch
import torch.nn as nn
import torch.nn.functional as F

# neural network (do research into this for OCR)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d()
        self.relu  = nn.ReLU()

    def forward(self):
        pass
