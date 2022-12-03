from typing import Optional
from typing import Tuple
from typing import Union

import torch

from .nets_utils import make_pad_mask


class Stft(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Union[int, None] = 512,
        hop_length: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        kaldi_padding_mode=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.kaldi_padding_mode = kaldi_padding_mode
        if self.kaldi_padding_mode:
            self.win_length = 400

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"pad_mode={self.pad_mode}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if not self.kaldi_padding_mode:
            output = torch.stft(
                input,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                pad_mode=self.pad_mode,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=False
            )
        else:
            # NOTE(sx): Use Kaldi-fasion padding, maybe wrong
            num_pads = self.n_fft - self.win_length
            input = torch.nn.functional.pad(input, (num_pads, 0))
            output = torch.stft(
                input,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=False,
                pad_mode=self.pad_mode,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=False
            )

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad
            olens = torch.div(ilens - self.win_length, self.hop_length, rounding_mode='floor') + 1
            # olens = ilens - self.win_length // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens
