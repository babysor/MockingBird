from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.nets_utils import make_pad_mask


class MaskedMSELoss(nn.Module):
    def __init__(self, frames_per_step):
        super().__init__()
        self.frames_per_step = frames_per_step
        self.mel_loss_criterion = nn.MSELoss(reduction='none')
        # self.loss = nn.MSELoss()
        self.stop_loss_criterion = nn.BCEWithLogitsLoss(reduction='none')   

    def get_mask(self, lengths, max_len=None):
        # lengths: [B,]
        if max_len is None:
            max_len = torch.max(lengths)
        batch_size = lengths.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(lengths.device)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()

    def forward(self, mel_pred, mel_pred_postnet, mel_trg, lengths, 
                stop_target, stop_pred):
        ## process stop_target
        B = stop_target.size(0)
        stop_target = stop_target.reshape(B, -1, self.frames_per_step)[:, :, 0]
        stop_lengths = torch.ceil(lengths.float() / self.frames_per_step).long()
        stop_mask = self.get_mask(stop_lengths, int(mel_trg.size(1)/self.frames_per_step))

        mel_trg.requires_grad = False
        # (B, T, 1)
        mel_mask = self.get_mask(lengths, mel_trg.size(1)).unsqueeze(-1)
        # (B, T, D)
        mel_mask = mel_mask.expand_as(mel_trg)
        mel_loss_pre = (self.mel_loss_criterion(mel_pred, mel_trg) * mel_mask).sum() / mel_mask.sum()
        mel_loss_post = (self.mel_loss_criterion(mel_pred_postnet, mel_trg) * mel_mask).sum() / mel_mask.sum()
        
        mel_loss = mel_loss_pre + mel_loss_post

        # stop token loss
        stop_loss = torch.sum(self.stop_loss_criterion(stop_pred, stop_target) * stop_mask) / stop_mask.sum()
        
        return mel_loss, stop_loss
