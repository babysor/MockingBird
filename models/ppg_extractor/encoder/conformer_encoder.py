#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from .convolution import ConvolutionModule
from .encoder_layer import EncoderLayer
from ..nets_utils import get_activation, make_pad_mask
from .vgg import VGG2L
from .attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .embedding import PositionalEncoding, ScaledPositionalEncoding, RelPositionalEncoding
from .layer_norm import LayerNorm
from .multi_layer_conv import Conv1dLinear, MultiLayeredConv1d
from .positionwise_feed_forward import PositionwiseFeedForward
from .repeat import repeat
from .subsampling import Conv2dNoSubsampling, Conv2dSubsampling


class ConformerEncoder(torch.nn.Module):
    """Conformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_pos_enc_layer_type: encoder positional encoding layer type
    :param str encoder_attn_layer_type: encoder attention layer type
    :param str activation_type: encoder activation function type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        no_subsample=False,
        subsample_by_2=False,
    ):
        """Construct an Encoder object."""
        super().__init__()
        
        self._output_size = attention_dim
        idim = input_size

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            logging.info("Encoder input layer type: conv2d")
            if no_subsample:
                self.embed = Conv2dNoSubsampling(
                    idim,
                    attention_dim,
                    dropout_rate,
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
            else:
                self.embed = Conv2dSubsampling(
                    idim,
                    attention_dim,
                    dropout_rate,
                    pos_enc_class(attention_dim, positional_dropout_rate),
                    subsample_by_2,  # NOTE(Sx): added by songxiang
                )
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
    
    def output_size(self) -> int:
        return self._output_size 
    
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input lengths (B)
            prev_states: Not to be used now.
        Returns:
            Position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(self.embed, (Conv2dSubsampling, Conv2dNoSubsampling, VGG2L)):
            # print(xs_pad.shape)
            xs_pad, masks = self.embed(xs_pad, masks)
            # print(xs_pad[0].size())
        else:
            xs_pad = self.embed(xs_pad)
        xs_pad, masks = self.encoders(xs_pad, masks)
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
    
    # def forward(self, xs, masks):
        # """Encode input sequence.

        # :param torch.Tensor xs: input tensor
        # :param torch.Tensor masks: input mask
        # :return: position embedded tensor and mask
        # :rtype Tuple[torch.Tensor, torch.Tensor]:
        # """
        # if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            # xs, masks = self.embed(xs, masks)
        # else:
            # xs = self.embed(xs)

        # xs, masks = self.encoders(xs, masks)
        # if isinstance(xs, tuple):
            # xs = xs[0]

        # if self.normalize_before:
            # xs = self.after_norm(xs)
        # return xs, masks
