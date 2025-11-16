import habana_frameworks.torch
import time
import logging
from pathlib import Path
from models.synthesizer.inference import Synthesizer
from models.encoder import inference as encoder
from models.vocoder.hifigan import inference as gan_vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import sys
import os
import re
import cn2an

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

vocoder = gan_vocoder

def create_input_file(txt_file_name):
    with open(txt_file_name, "w") as f:
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")
        f.write("你好，我是一个AI\n")
        f.write("我是一个AI，你好\n")

def gen_one_wav(synthesizer, vocoder, in_fpath, embed, texts, file_name, seq, device):
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
    filename = "%s_%d_%s.wav" % ("data/output", time.time(), device)
    sf.write(filename, generated_wav, synthesizer.sample_rate)

    print("\nSaved output as %s\n\n" % filename)
    
    
def generate_wav(encoder, synthesizer, vocoder, in_fpath, input_txt, file_name, device): 
    start = time.perf_counter()
    
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
        gen_one_wav(synthesizer, vocoder, in_fpath, embed, processed_texts, file_name, seq, device)
        processed_texts = []
        cur_num = 0
    
    if len(processed_texts)>0:
      seq = seq +1
      gen_one_wav(synthesizer, vocoder, in_fpath, embed, processed_texts, file_name, seq, device)
    
    end = time.perf_counter()
    return end - start
    
def run(encoder, synthesizer, vocoder, device: str = None):    
    my_txt = ""
    
    txt_file_name = Path("data/input.txt")
    wav_file_name = Path("data/samples/T0055G0013S0005.wav")
    
    if not txt_file_name.exists():
        print(f"Input file {txt_file_name} does not exist. Creating a sample input file.")
        create_input_file(txt_file_name)
    
    with open(txt_file_name, "r") as f:
        for line in f.readlines():
            my_txt += line

    output = cn2an.transform(my_txt, "an2cn")
    print(output)
    runtime = generate_wav(
        encoder,
        synthesizer,
        vocoder, 
        wav_file_name, 
        output, 
        txt_file_name, 
        device,
    )
    logger.info(f"Execution time: {runtime:.4f} seconds")
    return runtime

def run_n_times(n: int, device: str = None):
    ENC_MODEL_FPATH = Path("data/ckpt/encoder/pretrained.pt")
    SYN_MODEL_FPATH = Path("data/ckpt/synthesizer/pretrained/mandarin.pt")
    VOC_MODEL_FPATH = Path("data/ckpt/vocoder/pretrained/g_hifigan.pt")
    
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(ENC_MODEL_FPATH, device=device)
    synthesizer = Synthesizer(SYN_MODEL_FPATH, device=device)
    vocoder.load_model(VOC_MODEL_FPATH, device=device)
    
    times = []
    
    for i in range(n):
        runtime = run(encoder, synthesizer, vocoder, device)
        times.append(runtime)
    return np.mean(times)

if __name__ == "__main__":
    logger.debug("Running on HPU")
    hpu_time = run_n_times(3, "hpu")
    
    logger.debug("Running on CPU")
    cpu_time = run_n_times(3, "cpu")
    
    logger.info(f"HPU time: {hpu_time:.4f} seconds")
    logger.info(f"CPU time: {cpu_time:.4f} seconds")

