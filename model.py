from pathlib import Path
import sys

import torch
import torch.nn as nn

from tcn import TemporalConvNet

class SARPixelTCN(nn.Module):
    """
    Causal TCN for pixel-wise SAR next-step prediction.

    Input:
        x: [B, T, 1]
    Output:
        pred: [B, 1, T]
        features: [B, C, T]
    """

    def __init__(self, num_channels, kernel_size=2, dropout=0.1, input_channels=1):
        super().__init__()
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.input_channels = input_channels

        self.encoder = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=self.num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        hidden_channels = self.num_channels[-1]
        self.prediction_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        print(f"Finish model init | input_channels={input_channels}")

    def forward(self, x):
        # Dataset/DataLoader provide x as [B, T, 1].
        x = x.permute(0, 2, 1).contiguous()  # [B, 1, T]
        features = self.encoder(x)  # [B, C, T]
        pred = self.prediction_head(features)  # [B, 1, T]
        return pred, features

    def encode(self, x):
        # x: [B, T, 1] -> features: [B, C, T]
        x = x.permute(0, 2, 1).contiguous()
        return self.encoder(x)


def model_config(model):
    return {
        "num_channels": model.num_channels,
        "kernel_size": model.kernel_size,
        "dropout": model.dropout,
        "input_channels": model.input_channels,
    }

