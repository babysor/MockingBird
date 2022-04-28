import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import Linear, Conv1d


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, num_mels=80,
                 num_layers=5,
                 hidden_dim=512,
                 kernel_size=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    num_mels, hidden_dim,
                    kernel_size=kernel_size, stride=1,
                    padding=int((kernel_size - 1) / 2),
                    dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hidden_dim)))

        for i in range(1, num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size, stride=1,
                        padding=int((kernel_size - 1) / 2),
                        dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hidden_dim)))

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    hidden_dim, num_mels,
                    kernel_size=kernel_size, stride=1,
                    padding=int((kernel_size - 1) / 2),
                    dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mels)))

    def forward(self, x):
        # x: (B, num_mels, T_dec)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
