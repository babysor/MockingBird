#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import logging
import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length or 1/2 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, 
                 subsample_by_2=False,
        ):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.subsample_by_2 = subsample_by_2
        if subsample_by_2:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, kernel_size=5, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(odim * (idim // 2), odim),
                pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(odim * (idim // 4), odim),
                pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
            )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        if self.subsample_by_2:
            return x, x_mask[:, :, ::2]
        else:
            return x, x_mask[:, :, ::2][:, :, ::2]

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dNoSubsampling(torch.nn.Module):
    """Convolutional 2D without subsampling.

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        logging.info("Encoder does not do down-sample on mel-spectrogram.")
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * idim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
