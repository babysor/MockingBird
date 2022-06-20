
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile
import resampy

from ppg_extractor import load_model
import encoder.inference as Encoder
from encoder.audio import preprocess_wav
from encoder import audio
from utils.f0_utils import compute_f0

from torch.multiprocessing import Pool, cpu_count
from functools import partial

SAMPLE_RATE=16000

def _compute_bnf(
    wav: any,
    output_fpath: str,
    device: torch.device,
    ppg_model_local: any,
):
    """
    Compute CTC-Attention Seq2seq ASR encoder bottle-neck features (BNF).
    """
    ppg_model_local.to(device)
    wav_tensor = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    wav_length = torch.LongTensor([wav.shape[0]]).to(device)
    with torch.no_grad():
        bnf = ppg_model_local(wav_tensor, wav_length) 
    bnf_npy = bnf.squeeze(0).cpu().numpy()
    np.save(output_fpath, bnf_npy, allow_pickle=False)
    return bnf_npy, len(bnf_npy)

def _compute_f0_from_wav(wav, output_fpath):
    """Compute merged f0 values."""
    f0 = compute_f0(wav, SAMPLE_RATE)
    np.save(output_fpath, f0, allow_pickle=False)
    return f0, len(f0)

def _compute_spkEmbed(wav, output_fpath, encoder_model_local, device):
    Encoder.set_model(encoder_model_local)
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = Encoder.compute_partial_slices(len(wav), rate=1.3, min_pad_coverage=0.75)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = Encoder.embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    np.save(output_fpath, embed, allow_pickle=False)
    return embed, len(embed)

def preprocess_one(wav_path, out_dir, device, ppg_model_local, encoder_model_local):
    # wav = preprocess_wav(wav_path)
    # try:
    wav, sr = soundfile.read(wav_path)
    if len(wav) < sr:
        return None, sr, len(wav)
    if sr != SAMPLE_RATE:
        wav = resampy.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    utt_id = os.path.basename(wav_path).rstrip(".wav")

    _, length_bnf = _compute_bnf(output_fpath=f"{out_dir}/bnf/{utt_id}.ling_feat.npy", wav=wav, device=device, ppg_model_local=ppg_model_local)
    _, length_f0 = _compute_f0_from_wav(output_fpath=f"{out_dir}/f0/{utt_id}.f0.npy", wav=wav)
    _, length_embed = _compute_spkEmbed(output_fpath=f"{out_dir}/embed/{utt_id}.npy",  device=device, encoder_model_local=encoder_model_local, wav=wav)

def preprocess_dataset(datasets_root, dataset, out_dir, n_processes, ppg_encoder_model_fpath, speaker_encoder_model):
    # Glob wav files
    wav_file_list = sorted(Path(f"{datasets_root}/{dataset}").glob("**/*.wav"))
    print(f"Globbed {len(wav_file_list)} wav files.")

    out_dir.joinpath("bnf").mkdir(exist_ok=True, parents=True)
    out_dir.joinpath("f0").mkdir(exist_ok=True, parents=True)
    out_dir.joinpath("embed").mkdir(exist_ok=True, parents=True)
    ppg_model_local = load_model(ppg_encoder_model_fpath, "cpu")
    encoder_model_local = Encoder.load_model(speaker_encoder_model, "cpu")
    if n_processes is None:
        n_processes = cpu_count()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    func = partial(preprocess_one, out_dir=out_dir, ppg_model_local=ppg_model_local, encoder_model_local=encoder_model_local, device=device)
    job = Pool(n_processes).imap(func, wav_file_list)
    list(tqdm(job, "Preprocessing", len(wav_file_list), unit="wav"))

    # finish processing and mark
    t_fid_file = out_dir.joinpath("train_fidlist.txt").open("w", encoding="utf-8")
    d_fid_file = out_dir.joinpath("dev_fidlist.txt").open("w", encoding="utf-8")
    e_fid_file = out_dir.joinpath("eval_fidlist.txt").open("w", encoding="utf-8")
    for file in sorted(out_dir.joinpath("f0").glob("*.npy")):
        id = os.path.basename(file).split(".f0.npy")[0]
        if id.endswith("01"):
            d_fid_file.write(id + "\n")
        elif id.endswith("09"):
            e_fid_file.write(id + "\n")
        else:
            t_fid_file.write(id + "\n")
    t_fid_file.close()
    d_fid_file.close()
    e_fid_file.close()
    return len(wav_file_list)
