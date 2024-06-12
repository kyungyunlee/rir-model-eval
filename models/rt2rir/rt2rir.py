import numpy as np
from models.model_utils import (
    get_decayfitnet_params,
    octave_filter,
    get_fadeout_fadein_window,
)
from DecayFitNet.python.toolbox.core import decay_kernel, schroeder_to_envelope


def generate(rir, sr, filter_frequencies):
    T, A, noise, edc_norm_vals = get_decayfitnet_params(rir, sr, filter_frequencies)
    avg_rt60 = np.mean(T[2:5])
    # BINAURALIZATION OF OMNIDIRECTIONAL ROOM IMPULSE RESPONSES - ALGORITHM AND TECHNICAL EVALUATION
    mixing_time = int(0.05 * sr) if avg_rt60 <= 0.5 else int(0.1 * sr)
    target_rir_length = len(rir)

    time_axis = np.linspace(0, (target_rir_length - 1) / sr, target_rir_length)

    envelopes_T, envelopes_A = schroeder_to_envelope(T, A, sr)

    envelopes = decay_kernel(envelopes_T, time_axis)
    envelopes = envelopes[:, :-1] * envelopes_A

    noise = np.random.randn(target_rir_length)
    filtered_noise = octave_filter(noise, sr, filter_frequencies)

    generated_rir = np.zeros_like(noise)

    for k in range(len(filter_frequencies)):

        octave_filtered_noise = filtered_noise[k, :]
        rms_val = np.sqrt(np.mean(octave_filtered_noise**2))
        # Setting the RMS value of the filtered noise to 1 again.
        octave_filtered_noise = octave_filtered_noise / rms_val
        # Apply the original envelope
        h = octave_filtered_noise * envelopes[:, k]

        generated_rir += h

    # Match energy level
    late_energy_original = np.sqrt(np.mean(rir[mixing_time:] ** 2))
    late_energy_generated = np.sqrt(np.mean(generated_rir[mixing_time:] ** 2))
    generated_rir = generated_rir / late_energy_generated * late_energy_original

    # Combine with direct , ealry
    window_length = int(sr * 0.005)
    # window_length = int(sr * 0.01)
    direct_early_window, late_window = get_fadeout_fadein_window(
        len(rir), window_length, mixing_time
    )
    original_direct_early = rir * direct_early_window

    # Add the generated late part
    generated_late = generated_rir * late_window

    full_generated_rir = original_direct_early + generated_late

    return full_generated_rir
