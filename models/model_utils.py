import numpy as np
import scipy.signal as signal
from DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from DecayFitNet.python.toolbox.core import (
    discard_last_n_percent,
    decay_model,
    PreprocessRIR,
)


def get_decayfitnet_params(rir, fs, filter_frequencies):
    n_slopes = 1

    # Prepare the model
    decayfitnet = DecayFitNetToolbox(
        n_slopes=n_slopes, sample_rate=fs, filter_frequencies=filter_frequencies
    )

    # Process: analyse_full_rir: if RIR onset should be detected, set this to False
    estimated_parameters_decayfitnet, norm_vals = decayfitnet.estimate_parameters(
        rir, analyse_full_rir=True
    )
    [T, A, noise] = estimated_parameters_decayfitnet
    T = T[:, 0]
    A = A[:, 0]
    noise = noise[:, 0]

    return T, A, noise, norm_vals


def find_direct_location(rir):
    peak_val = np.max(np.abs(rir))
    peak_val_location = np.argmax(np.abs(rir))
    threshold_val = 10 ** (-1 / 2) * peak_val
    candidate_list = np.argwhere(np.array(rir[:peak_val_location]) > threshold_val)
    if len(candidate_list) == 0:
        return peak_val_location
    else:
        direct_location = candidate_list[0][0]
        return direct_location


def process_rir(rir, sr):
    # Cut out or pad beginning part if needed
    # 20 ms
    frontal_silence_length = int(sr * 0.003)
    direct_location = find_direct_location(rir)
    print(direct_location, frontal_silence_length)

    if direct_location < frontal_silence_length:
        # pad
        print("pad")
        pad_length = frontal_silence_length - direct_location
        rir_processed = np.zeros((len(rir) + pad_length,))
        rir_processed[pad_length:] = rir
    else:
        # trim
        print("trim")
        rir_processed = rir[direct_location - frontal_silence_length :]

    # Peak 1
    rir_processed = rir_processed / np.max(np.abs(rir_processed))

    return rir_processed


def process_brir(brir, sr):
    # brir : (2, len)
    frontal_silence_length = int(sr * 0.003)
    direct_location_left = find_direct_location(brir[0])
    direct_location_right = find_direct_location(brir[1])

    if direct_location_left < direct_location_right:
        direct_location = direct_location_left
    else:
        direct_location = direct_location_right

    if direct_location < frontal_silence_length:
        # pad
        print("pad")
        pad_length = frontal_silence_length - direct_location
        rir_processed = np.zeros((2, brir.shape[1] + pad_length))
        rir_processed[:, pad_length:] = brir
        length_diff = pad_length  # Positive
    else:
        # trim
        print("trim")
        rir_processed = brir[:, direct_location - frontal_silence_length :]
        length_diff = -(direct_location - frontal_silence_length)  # Negative

    rir_processed = rir_processed / np.max(np.abs(rir_processed))
    return rir_processed, length_diff


def get_direct_window(rir, sr):
    direct_location = find_direct_location(rir)
    window_length = int(sr * 0.005)
    _window = np.hanning(window_length)
    window = np.zeros((len(rir),))
    window[:direct_location] = 1
    window[direct_location : direct_location + window_length // 2] = _window[
        window_length // 2 :
    ]

    return window, window_length


def get_fadeout_fadein_window(signal_length, window_length, mixing_location):
    hanning_window = np.hanning(window_length)
    _fadein_window = hanning_window[: window_length // 2]
    _fadeout_window = hanning_window[window_length // 2 :]

    fadeout_window = np.zeros((signal_length,))
    fadein_window = np.zeros((signal_length,))

    fadeout_window[:mixing_location] = 1
    fadeout_window[mixing_location : mixing_location + window_length // 2] = (
        _fadeout_window
    )

    fadein_window[mixing_location:] = 1
    fadein_window[mixing_location - window_length // 2 : mixing_location] = (
        _fadein_window
    )

    return fadeout_window, fadein_window


def octave_filter(y, fs, freq):
    order = 5
    # octave = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    filtered_y = []
    for f in freq:
        f_cutoff = f * np.array([1 / np.sqrt(2), np.sqrt(2)])
        # print (f_cutoff)
        sos = signal.butter(
            N=order, Wn=f_cutoff, btype="bandpass", fs=fs, analog=False, output="sos"
        )

        filtered_y.append(signal.sosfilt(sos, y))
    return np.array(filtered_y)
