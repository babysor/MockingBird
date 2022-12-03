import copy
from typing import Tuple
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor

from .log_mel import LogMel
from .stft import Stft


class DefaultFrontend(torch.nn.Module):
    """Conventional frontend structure for ASR

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: 16000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 160,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        norm=1,
        frontend_conf=None, #Optional[dict] = get_default_kwargs(Frontend),
        kaldi_padding_mode=False,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.downsample_rate = downsample_rate

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            kaldi_padding_mode=kaldi_padding_mode
        )
        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm,
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])

        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
               
        # NOTE(sx): pad
        max_len = input_feats.size(1)
        if self.downsample_rate > 1 and max_len % self.downsample_rate != 0:
            padding = self.downsample_rate - max_len % self.downsample_rate
            # print("Logmel: ", input_feats.size())
            input_feats = torch.nn.functional.pad(input_feats, (0, 0, 0, padding),
                                                  "constant", 0)
            # print("Logmel(after padding): ",input_feats.size())
            feats_lens[torch.argmax(feats_lens)] = max_len + padding 

        return input_feats, feats_lens
