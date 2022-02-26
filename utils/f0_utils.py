import logging
import numpy as np
import pyworld
from scipy.interpolate import interp1d
from scipy.signal import firwin, get_window, lfilter

def compute_mean_std(lf0):
    nonzero_indices = np.nonzero(lf0)
    mean = np.mean(lf0[nonzero_indices])
    std = np.std(lf0[nonzero_indices])
    return mean, std 


def compute_f0(wav, sr=16000, frame_period=10.0):
    """Compute f0 from wav using pyworld harvest algorithm."""
    wav = wav.astype(np.float64)
    f0, _ = pyworld.harvest(
        wav, sr, frame_period=frame_period, f0_floor=80.0, f0_ceil=600.0)
    return f0.astype(np.float32)

def f02lf0(f0):
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0

def get_converted_lf0uv(
    wav, 
    lf0_mean_trg, 
    lf0_std_trg,
    convert=True,
):
    f0_src = compute_f0(wav)
    if not convert:
        uv, cont_lf0 = get_cont_lf0(f0_src)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        return lf0_uv

    lf0_src = f02lf0(f0_src)
    lf0_mean_src, lf0_std_src = compute_mean_std(lf0_src)
    
    lf0_vc = lf0_src.copy()
    lf0_vc[lf0_src > 0.0] = (lf0_src[lf0_src > 0.0] - lf0_mean_src) / lf0_std_src * lf0_std_trg + lf0_mean_trg
    f0_vc = lf0_vc.copy()
    f0_vc[lf0_src > 0.0] = np.exp(lf0_vc[lf0_src > 0.0])
    
    uv, cont_lf0_vc = get_cont_lf0(f0_vc)
    lf0_uv = np.concatenate([cont_lf0_vc[:, np.newaxis], uv[:, np.newaxis]], axis=1)
    return lf0_uv

def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_cont_lf0(f0, frame_period=10.0, lpf=False):
    uv, cont_f0 = convert_continuos_f0(f0)
    if lpf:
        cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (frame_period * 0.001)), cutoff=20)
        cont_lf0_lpf = cont_f0_lpf.copy()
        nonzero_indices = np.nonzero(cont_lf0_lpf)
        cont_lf0_lpf[nonzero_indices] = np.log(cont_f0_lpf[nonzero_indices])
        # cont_lf0_lpf = np.log(cont_f0_lpf)
        return uv, cont_lf0_lpf 
    else:
        nonzero_indices = np.nonzero(cont_f0)
        cont_lf0 = cont_f0.copy()
        cont_lf0[cont_f0>0] = np.log(cont_f0[cont_f0>0])
        return uv, cont_lf0
