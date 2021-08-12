import torch.nn as nn
import torch
from SENet.Test_import import add


class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


a = torch.rand(1, 256, 5, 5)
b = nn.AdaptiveAvgPool2d(1)(a)
c = b.view(1, 256)
se_layer = SElayer(256)
d = se_layer.fc(c)
e = d.view(1, 256, 1, 1)
f = e.expand_as(a)
g = a * f
print(a)

print(add(9, 9))
