import time
import os
import argparse
import torch
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from ppg_extractor import load_model
import librosa
import soundfile as sf
from utils.load_yaml import HpsYaml

from encoder.audio import preprocess_wav
from encoder import inference as speacker_encoder
from vocoder.hifigan import inference as vocoder
from ppg2mel import MelDecoderMOLv2
from utils.f0_utils import compute_f0, f02lf0, compute_mean_std, get_converted_lf0uv


def _build_ppg2mel_model(model_config, model_file, device):
    ppg2mel_model = MelDecoderMOLv2(
        **model_config["model"]
    ).to(device)
    ckpt = torch.load(model_file, map_location=device)
    ppg2mel_model.load_state_dict(ckpt["model"])
    ppg2mel_model.eval()
    return ppg2mel_model


@torch.no_grad()
def convert(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    step = os.path.basename(args.ppg2mel_model_file)[:-4].split("_")[-1]

    # Build models
    print("Load PPG-model, PPG2Mel-model, Vocoder-model...")
    ppg_model = load_model(
        Path('./ppg_extractor/saved_models/24epoch.pt'),
        device,
    )
    ppg2mel_model = _build_ppg2mel_model(HpsYaml(args.ppg2mel_model_train_config), args.ppg2mel_model_file, device) 
    # vocoder.load_model('./vocoder/saved_models/pretrained/g_hifigan.pt', "./vocoder/hifigan/config_16k_.json")
    vocoder.load_model('./vocoder/saved_models/24k/g_02830000.pt')
    # Data related
    ref_wav_path = args.ref_wav_path
    ref_wav = preprocess_wav(ref_wav_path)
    ref_fid = os.path.basename(ref_wav_path)[:-4]
    
    # TODO: specify encoder
    speacker_encoder.load_model(Path("encoder/saved_models/pretrained_bak_5805000.pt"))
    ref_spk_dvec = speacker_encoder.embed_utterance(ref_wav)
    ref_spk_dvec = torch.from_numpy(ref_spk_dvec).unsqueeze(0).to(device)
    ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
    
    source_file_list = sorted(glob.glob(f"{args.wav_dir}/*.wav"))
    print(f"Number of source utterances: {len(source_file_list)}.")
    
    total_rtf = 0.0
    cnt = 0
    for src_wav_path in tqdm(source_file_list):
        # Load the audio to a numpy array:
        src_wav, _ = librosa.load(src_wav_path, sr=16000)
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).float().to(device)
        src_wav_lengths = torch.LongTensor([len(src_wav)]).to(device)
        ppg = ppg_model(src_wav_tensor, src_wav_lengths)

        lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
        min_len = min(ppg.shape[1], len(lf0_uv))

        ppg = ppg[:, :min_len]
        lf0_uv = lf0_uv[:min_len]
        
        start = time.time()
        _, mel_pred, att_ws = ppg2mel_model.inference(
            ppg,
            logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device),
            spembs=ref_spk_dvec,
        )
        src_fid = os.path.basename(src_wav_path)[:-4]
        wav_fname = f"{output_dir}/vc_{src_fid}_ref_{ref_fid}_step{step}.wav"
        mel_len = mel_pred.shape[0]
        rtf = (time.time() - start) / (0.01 * mel_len)
        total_rtf += rtf
        cnt += 1
        # continue
        mel_pred= mel_pred.transpose(0, 1)
        y, output_sample_rate = vocoder.infer_waveform(mel_pred.cpu())
        sf.write(wav_fname, y.squeeze(), output_sample_rate, "PCM_16")
    
    print("RTF:")
    print(total_rtf / cnt)


def get_parser():
    parser = argparse.ArgumentParser(description="Conversion from wave input")
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        required=True,
        help="Source wave directory.",
    )
    parser.add_argument(
        "--ref_wav_path",
        type=str,
        required=True,
        help="Reference wave file path.",
    )
    parser.add_argument(
        "--ppg2mel_model_train_config", "-c",
        type=str,
        default=None,
        required=True,
        help="Training config file (yaml file)",
    )
    parser.add_argument(
        "--ppg2mel_model_file", "-m",
        type=str,
        default=None,
        required=True,
        help="ppg2mel model checkpoint file path"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="vc_gens_vctk_oneshot",
        help="Output folder to save the converted wave."
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    convert(args)

if __name__ == "__main__":
    main()
