import argparse
import torch
from pathlib import Path
import yaml

from .frontend import DefaultFrontend
from .utterance_mvn import UtteranceMVN
from .encoder.conformer_encoder import ConformerEncoder

_model = None # type: PPGModel
_device = None

class PPGModel(torch.nn.Module):
    def __init__(
        self,
        frontend,
        normalizer,
        encoder,
    ):
        super().__init__()
        self.frontend = frontend
        self.normalize = normalizer
        self.encoder = encoder

    def forward(self, speech, speech_lengths):
        """

        Args:
            speech (tensor): (B, L)
            speech_lengths (tensor): (B, )

        Returns:
            bottle_neck_feats (tensor): (B, L//hop_size, 144)

        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        feats, feats_lengths = self.normalize(feats, feats_lengths)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        return encoder_out

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ):
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths
        
    def extract_from_wav(self, src_wav):
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).float().to(_device)
        src_wav_lengths = torch.LongTensor([len(src_wav)]).to(_device)
        return self(src_wav_tensor, src_wav_lengths)


def build_model(args):
    normalizer = UtteranceMVN(**args.normalize_conf)
    frontend = DefaultFrontend(**args.frontend_conf)
    encoder = ConformerEncoder(input_size=80, **args.encoder_conf)
    model = PPGModel(frontend, normalizer, encoder)
    
    return model


def load_model(model_file, device=None):
    global _model, _device
    
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = device
    # search a config file
    model_config_fpaths = list(model_file.parent.rglob("*.yaml"))
    config_file = model_config_fpaths[0]
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args = argparse.Namespace(**args)

    model = build_model(args)
    model_state_dict = model.state_dict()

    ckpt_state_dict = torch.load(model_file, map_location=_device)
    ckpt_state_dict = {k:v for k,v in ckpt_state_dict.items() if 'encoder' in k}

    model_state_dict.update(ckpt_state_dict)
    model.load_state_dict(model_state_dict)

    _model = model.eval().to(_device)
    return _model


