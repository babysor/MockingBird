import random
import numpy as np
import torch
from utils.f0_utils import get_cont_lf0
import resampy
from .audio_utils import MAX_WAV_VALUE, load_wav, mel_spectrogram
from librosa.util import normalize
import os


SAMPLE_RATE=16000

def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split()[0] for l in f if l.strip()]
    return fids   

class OneshotVcDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        vctk_ppg_dir: str,
        libri_ppg_dir: str,
        vctk_f0_dir: str,
        libri_f0_dir: str,
        vctk_wav_dir: str,
        libri_wav_dir: str,
        vctk_spk_dvec_dir: str,
        libri_spk_dvec_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        ppg_file_ext: str = "ling_feat.npy",
        f0_file_ext: str = "f0.npy",
        wav_file_ext: str = "wav",
    ):
        self.fid_list = read_fids(meta_file)
        self.vctk_ppg_dir = vctk_ppg_dir
        self.libri_ppg_dir = libri_ppg_dir
        self.vctk_f0_dir = vctk_f0_dir
        self.libri_f0_dir = libri_f0_dir
        self.vctk_wav_dir = vctk_wav_dir
        self.libri_wav_dir = libri_wav_dir
        self.vctk_spk_dvec_dir = vctk_spk_dvec_dir
        self.libri_spk_dvec_dir = libri_spk_dvec_dir

        self.ppg_file_ext = ppg_file_ext
        self.f0_file_ext = f0_file_ext
        self.wav_file_ext = wav_file_ext

        self.min_max_norm_mel = min_max_norm_mel
        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def get_spk_dvec(self, fid):
        spk_name = fid
        if spk_name.startswith("p"):
            spk_dvec_path = f"{self.vctk_spk_dvec_dir}{os.sep}{spk_name}.npy"
        else:
            spk_dvec_path = f"{self.libri_spk_dvec_dir}{os.sep}{spk_name}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))
    
    def compute_mel(self, wav_path):
        audio, sr = load_wav(wav_path)
        if sr != SAMPLE_RATE:
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=SAMPLE_RATE,
            hop_size=160,
            win_size=1024,
            fmin=80,
            fmax=8000,
        )
        return melspec.squeeze(0).numpy().T

    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        if fid.startswith("p"):
            # vctk
            sub = fid.split("_")[0]
            ppg = np.load(f"{self.vctk_ppg_dir}{os.sep}{fid}.{self.ppg_file_ext}")
            f0 = np.load(f"{self.vctk_f0_dir}{os.sep}{fid}.{self.f0_file_ext}")
            mel = self.compute_mel(f"{self.vctk_wav_dir}{os.sep}{sub}{os.sep}{fid}.{self.wav_file_ext}")
        else:
            # aidatatang
            sub = fid[5:10]
            ppg = np.load(f"{self.libri_ppg_dir}{os.sep}{fid}.{self.ppg_file_ext}")
            f0 = np.load(f"{self.libri_f0_dir}{os.sep}{fid}.{self.f0_file_ext}")
            mel = self.compute_mel(f"{self.libri_wav_dir}{os.sep}{sub}{os.sep}{fid}.{self.wav_file_ext}")
        if self.min_max_norm_mel:
            mel = self.bin_level_min_max_norm(mel)
        
        f0, ppg, mel = self._adjust_lengths(f0, ppg, mel, fid)
        spk_dvec = self.get_spk_dvec(fid)

        # 2. Convert f0 to continuous log-f0 and u/v flags
        uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        # cont_lf0 = (cont_lf0 - np.amin(cont_lf0)) / (np.amax(cont_lf0) - np.amin(cont_lf0))
        # cont_lf0 = self.utt_mvn(cont_lf0)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)

        # uv, cont_f0 = convert_continuous_f0(f0)
        # cont_f0 = (cont_f0 - np.amin(cont_f0)) / (np.amax(cont_f0) - np.amin(cont_f0))
        # lf0_uv = np.concatenate([cont_f0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)
        
        return (ppg, lf0_uv, mel, spk_dvec, fid)

    def check_lengths(self, f0, ppg, mel, fid):
        LEN_THRESH = 10
        assert abs(len(ppg) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(ppg) - len(f0))}: for file {fid}"
        assert abs(len(mel) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(mel) - len(f0))}: for file {fid}"
    
    def _adjust_lengths(self, f0, ppg, mel, fid):
        self.check_lengths(f0, ppg, mel, fid)
        min_len = min(
            len(f0),
            len(ppg),
            len(mel),
        )
        f0 = f0[:min_len]
        ppg = ppg[:min_len]
        mel = mel[:min_len]
        return f0, ppg, mel

class MultiSpkVcCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, give_uttids=False,
                 f02ppg_length_ratio=1, use_spk_dvec=False):
        self.n_frames_per_step = n_frames_per_step
        self.give_uttids = give_uttids
        self.f02ppg_length_ratio = f02ppg_length_ratio
        self.use_spk_dvec = use_spk_dvec

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        ppgs = [x[0] for x in batch]
        lf0_uvs = [x[1] for x in batch]
        mels = [x[2] for x in batch]
        fids = [x[-1] for x in batch]
        if len(batch[0]) == 5:
            spk_ids = [x[3] for x in batch]
            if self.use_spk_dvec:
                # use d-vector
                spk_ids = torch.stack(spk_ids).float()
            else:
                # use one-hot ids
                spk_ids = torch.LongTensor(spk_ids)
        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        mel_lengths = [x.shape[0] for x in mels]
        max_ppg_len = max(ppg_lengths)
        max_mel_len = max(mel_lengths)
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += (self.n_frames_per_step - max_mel_len % self.n_frames_per_step)
        ppg_dim = ppgs[0].shape[1]
        mel_dim = mels[0].shape[1]
        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        mels_padded = torch.FloatTensor(batch_size, max_mel_len, mel_dim).zero_()
        lf0_uvs_padded = torch.FloatTensor(batch_size, self.f02ppg_length_ratio * max_ppg_len, 2).zero_()
        stop_tokens = torch.FloatTensor(batch_size, max_mel_len).zero_()
        for i in range(batch_size):
            cur_ppg_len = ppgs[i].shape[0]
            cur_mel_len = mels[i].shape[0]
            ppgs_padded[i, :cur_ppg_len, :] = ppgs[i]
            lf0_uvs_padded[i, :self.f02ppg_length_ratio*cur_ppg_len, :] = lf0_uvs[i]
            mels_padded[i, :cur_mel_len, :] = mels[i]
            stop_tokens[i, cur_ppg_len-self.n_frames_per_step:] = 1
        if len(batch[0]) == 5:
            ret_tup = (ppgs_padded, lf0_uvs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), spk_ids, stop_tokens)
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup
        else:
            ret_tup = (ppgs_padded, lf0_uvs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), stop_tokens)
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup
