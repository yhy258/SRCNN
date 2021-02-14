import torch.nn as nn

class SRCNN(nn.Module):
  def __init__(self):
    super(SRCNN, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(3, 64, 9, padding = 4),
        nn.ReLU(),
        nn.Conv2d(64,32,1, padding = 0),
        nn.ReLU(),
        nn.Conv2d(32,3, 5, padding = 2)
    )
  def forward(self, x):
    x = self.conv(x)
    return x