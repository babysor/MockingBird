#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer modules for FFT block in FastSpeech (Feed-forward Transformer)."""

import torch


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))
