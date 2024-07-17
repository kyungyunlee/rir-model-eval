import numpy as np 
from DecayFitNet.python.toolbox.core import  decay_kernel, schroeder_to_envelope
from models.model_utils import get_decayfitnet_params, find_direct_location, get_fadeout_fadein_window, octave_filter, get_direct_window


FS = 48000
WINDOW_LENGTH = 0.02
WINDOW_LENGTH_SAMPLES = int(FS * WINDOW_LENGTH)

MIXING_WINDOW_SIZE = 0.005
MIXING_WINDOW_SIZE_SAMPLES = int(MIXING_WINDOW_SIZE * FS)

FILTER_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]



def generate(rir) :
    initial_direct_location = find_direct_location(rir)

    ##### DIRECT EARLY 
    echo_density = [] 

    mixing_time = -1 
    for i in range(len(rir)): 
        start = max(0, i - WINDOW_LENGTH_SAMPLES//2)
        end = min(i + WINDOW_LENGTH_SAMPLES//2, len(rir))
        curr_signal = rir[start:end]
        hanning_window_size = int(len(curr_signal))
        hanning_window = np.hanning(hanning_window_size)

        if len(curr_signal) < WINDOW_LENGTH_SAMPLES:
            if start < len(curr_signal)//2:
                hanning_window = hanning_window[ -len(curr_signal):]
            else : 
                hanning_window = hanning_window[:len(curr_signal)]
        
        hanning_window /= np.sum(hanning_window)
        
        window_std = np.sqrt(np.sum(hanning_window * (curr_signal ** 2)))

        curr_echo_density = (1 / 0.3173) * np.sum(hanning_window * ((np.abs(curr_signal) > window_std) * 1))
        echo_density.append(curr_echo_density) 

         # Get mixing time 
        if curr_echo_density >= 1.0 and mixing_time < 0: 
            mixing_time = i
            
    echo_density = np.array(echo_density)
    assert len(rir) == len(echo_density)

    # Absolute echo density
    bandwidth = 2000
    echo_duration = 1 / bandwidth 
    print (mixing_time)

    N = mixing_time + int( MIXING_WINDOW_SIZE * FS )

    absolute_echo_density = [] 
    for i in range(N):
        curr_val = echo_density[i]
        curr_val = min(0.999, curr_val)
        absolute_val = (curr_val / echo_duration) / (1 - curr_val)
        absolute_echo_density.append(absolute_val)
    absolute_echo_density = np.array(absolute_echo_density)


    ##### Synthesis 
    T, A, noise, norm_vals = get_decayfitnet_params(rir, FS, FILTER_FREQUENCIES)


    N = mixing_time + int( MIXING_WINDOW_SIZE * FS )
    synth_rir = np.zeros(N)
    # Get interval and AED of first echo
    start = int(np.ceil(np.random.exponential(1/absolute_echo_density[0])*FS))
    save_start = start
    pPrev = absolute_echo_density[0] 
    while start < N:
        # Grab a random sample of noise with variance based on p
        synth_rir[start] = np.random.normal(0, np.sqrt(1/pPrev)) 
        
        # Get next interval time and AED
        tau = int(np.ceil(np.random.exponential(1/absolute_echo_density[start])*FS))
        pPrev = absolute_echo_density[start]
        start = start + tau
        # print (start)

    synth_rir[save_start] = 0.999

    sinc_f = np.sinc(np.arange(-8, 8))

    synth_rir = np.convolve(synth_rir, sinc_f)

    # plt.plot(synth_rir_interpolated)
   
   

    #### LATE REVERB

    # Set up decay model
    rir_len = len(rir)
    time_axis = np.linspace(0, (rir_len - 1) / FS, rir_len)

    # Determine envelope from EDF model
    envelopes_T, envelopes_A = schroeder_to_envelope(T, A, FS)

    envelopes = decay_kernel(envelopes_T, time_axis)
    envelopes = envelopes[:, :-1] * envelopes_A
    
    noise = np.random.randn(rir_len)
    filtered_noise = octave_filter(noise, FS, FILTER_FREQUENCIES)
    
    late_reverb = np.zeros_like(noise) 

    for k in range(len(T)) : 
        
        octave_filtered_noise = filtered_noise[k] 
        # Setting the RMS value of the filtered noise to 1 again. 
        rms_val = np.sqrt(np.mean(octave_filtered_noise**2)) 
        octave_filtered_noise = octave_filtered_noise / rms_val 
        # Apply the original envelope
        h = octave_filtered_noise * envelopes[:, k]
        
        late_reverb += h 


    # Shift
    echo_density_direct_location = save_start
    shift_amount = initial_direct_location - echo_density_direct_location
    synth_rir_shifted = np.zeros_like(synth_rir)
    if shift_amount < 0 : 
        # need to shift left 
        synth_rir_shifted[:len(synth_rir) + shift_amount] = synth_rir[abs(shift_amount):]
    else :
        # need to shift right 
        synth_rir_shifted[shift_amount:] = synth_rir[:-shift_amount]

    synth_rir_shifted_full_length = np.zeros_like(rir)
    synth_rir_shifted_full_length[:len(synth_rir_shifted)] = synth_rir_shifted

    mixing_time += shift_amount

    # ADD Late reverb 
    mixing_window_length = int(FS* MIXING_WINDOW_SIZE)
    direct_early_window, late_window = get_fadeout_fadein_window(len(rir), mixing_window_length, mixing_time)

    result_rir = np.zeros((len(rir),))
    result_rir += synth_rir_shifted_full_length * direct_early_window
    result_rir += late_reverb * late_window


    # Match energy level 
    late_energy_original = np.sqrt(np.mean(rir[mixing_time:]**2))
    late_energy_generated = np.sqrt(np.mean(result_rir[mixing_time:]**2))
    result_rir = result_rir/late_energy_generated*late_energy_original

    # Replace direct 
    original_direct_window, _ = get_direct_window(rir, FS)
    original_direct = original_direct_window * rir

    echo_density_direct_window, _ = get_direct_window(result_rir, FS)
    echo_density_direct = echo_density_direct_window * result_rir 

    result_rir -= echo_density_direct
    result_rir += original_direct 
            
    return result_rir
