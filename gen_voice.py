from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder.wavernn import inference as rnn_vocoder
from vocoder.hifigan import inference as gan_vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
import re
import cn2an
import glob

from audioread.exceptions import NoBackendError
vocoder = gan_vocoder

def gen_one_wav(synthesizer, in_fpath, embed, texts, file_name, seq):
    embeds = [embed] * len(texts)
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds, style_idx=-1, min_stop_token=4, steps=400)
    #spec = specs[0]
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    # If seed is specified, reset torch seed and reload vocoder
    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav, output_sample_rate = vocoder.infer_waveform(spec)
    
    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [generated_wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * synthesizer.sample_rate))] * len(breaks)
    generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    
    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)
    generated_wav = generated_wav / np.abs(generated_wav).max() * 0.97
        
    # Save it on the disk
    model=os.path.basename(in_fpath)
    filename = "%s_%d_%s.wav" %(file_name, seq, model)
    sf.write(filename, generated_wav, synthesizer.sample_rate)

    print("\nSaved output as %s\n\n" % filename)
    
    
def generate_wav(enc_model_fpath, syn_model_fpath, voc_model_fpath, in_fpath, input_txt, file_name): 
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" % 
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(enc_model_fpath)
    synthesizer = Synthesizer(syn_model_fpath)
    vocoder.load_model(voc_model_fpath)

    encoder_wav = synthesizer.load_preprocess_wav(in_fpath)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    texts = input_txt.split("\n")
    seq=0
    each_num=1500
    
    punctuation = '！，。、,' # punctuate and split/clean text
    processed_texts = []
    cur_num = 0
    for text in texts:
      for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
        if processed_text:
            processed_texts.append(processed_text.strip())
            cur_num += len(processed_text.strip())
      if cur_num > each_num:
        seq = seq +1
        gen_one_wav(synthesizer, in_fpath, embed, processed_texts, file_name, seq)
        processed_texts = []
        cur_num = 0
    
    if len(processed_texts)>0:
      seq = seq +1
      gen_one_wav(synthesizer, in_fpath, embed, processed_texts, file_name, seq)

if (len(sys.argv)>=3):
    my_txt = ""
    print("reading from :", sys.argv[1])
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            #line = line.strip('\n')
            my_txt += line
    txt_file_name = sys.argv[1]
    wav_file_name = sys.argv[2]

    output = cn2an.transform(my_txt, "an2cn")
    print(output)
    generate_wav(
    Path("encoder/saved_models/pretrained.pt"),
    Path("synthesizer/saved_models/mandarin.pt"),
    Path("vocoder/saved_models/pretrained/g_hifigan.pt"), wav_file_name, output, txt_file_name
    )

else:
    print("please input the file name")
    exit(1)


