import torch
import torch.nn as nn
import torch.functional as F

class Network(nn.Module):
    def __init__(self, height: int = 28, width: int = 28):
        super(Network, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(32),
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Flatten(),
            nn.Linear(height * width, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
