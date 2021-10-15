import torch
import torch.nn as nn
import torch.nn.functional as F

# neural network (do research into this for OCR)
class Network(nn.Module):
    def __init__(self, in_channels: int = 4):
        super(Network, self).__init__()
        self.p = lambda x: (x[0] // 2, x[1] //2)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=self.p((3, 3))),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=self.p((2, 2))),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=self.p((3, 3))),
        )

    def forward(self):
        pass
