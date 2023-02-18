import os
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.utils.data

from utils.audio_utils import load_wav_to_torch, spectrogram
from utils.util import intersperse
from models.synthesizer.utils.text import text_to_sequence


"""Multi speaker version"""
class VitsDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audio_file_path, hparams):
        with open(audio_file_path, encoding='utf-8') as f:
            self.audio_metadata = [line.strip().split('|') for line in f]
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.datasets_root = hparams.datasets_root

        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audio_metadata)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audio_metadata_new = []
        lengths = []
        
        # for audiopath, sid, text in self.audio_metadata:
        for wav_fpath, mel_fpath, embed_path, wav_length, mel_frames, text, spkid in self.audio_metadata:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audio_metadata_new.append([wav_fpath, mel_fpath, embed_path, wav_length, mel_frames, text, spkid])
                lengths.append(os.path.getsize(f'{self.datasets_root}{os.sep}audio{os.sep}{wav_fpath}') // (2 * self.hop_length))
        self.audio_metadata = audio_metadata_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audio_metadata):
        # separate filename, speaker_id and text
        wav_fpath, text, sid = audio_metadata[0], audio_metadata[5], audio_metadata[6]
        text = self.get_text(text)

        spec, wav = self.get_audio(f'{self.datasets_root}{os.sep}audio{os.sep}{wav_fpath}')
        sid = self.get_sid(sid)
        emo = torch.FloatTensor(np.load(f'{self.datasets_root}{os.sep}emo{os.sep}{wav_fpath.replace("audio", "emo")}'))
        return (text, spec, wav, sid, emo)

    def get_audio(self, filename):
        # Load preprocessed wav npy instead of reading from wav file
        audio = torch.FloatTensor(np.load(filename))
        audio_norm = audio.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram(audio_norm, self.filter_length,self.hop_length, self.win_length,
                center=False)
            torch.save(spec, spec_filename)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0) # 在所有文本数值序列中的元素前后都补充一个0 - 不适用于中文
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audio_metadata[index])

    def __len__(self):
        return len(self.audio_metadata)


class VitsDatasetCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        emo = torch.FloatTensor(len(batch), 1024)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            emo[i, :] = row[4]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing, emo
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, emo


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
