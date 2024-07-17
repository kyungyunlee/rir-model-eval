import numpy as np
from scipy.signal import butter, sosfilt, zpk2sos, sosfreqz
import matplotlib.pyplot as plt

def rt2slope(rt60, fs):
    '''convert time in seconds of 60db decay to energy decay slope'''
    return -60/(rt60*fs)

def hertz2unit(hertz, fs):
    '''Convert frequency from cycles per second to normalized'''
    return np.divide(hertz, fs//2).tolist()

def hertz2rad(hertz, fs):
    '''Convert frequency from cycles per second to rad'''
    return np.divide(hertz, fs)*2*np.pi

def db2mag(ydb):
    return 10**(ydb/20)

def mag2db(ylin):
    return 20*np.log10(ylin)

def normalize_energy(x):
    ''' normalize energy of x to 1 '''
    energy = np.sum(np.power(np.abs(x),2))
    return np.divide(x , np.power(energy, 1/2))

def get_frequency_samples(num):
    '''
    get frequency samples (in radians) sampled at linearly spaced points along the unit circle
    Args    num (int): number of frequency samples
    Output  frequency samples in radians between [0, pi]
    '''
    angle = np.linspace(0, 1, num)
    abs = np.ones(num)
    return abs * np.exp(1j * angle * np.pi) 

def octave_filtering(input_signal, fs, f_bands):
    num_bands = len(f_bands)
    out_bands = np.zeros((len(input_signal), num_bands))

    for b_idx in range(num_bands):
        if f_bands[b_idx] == 0:
            f_cutoff = (1 / np.sqrt(1.5)) * f_bands[b_idx + 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), output='zpk')
        elif f_bands[b_idx] == fs / 2:
            f_cutoff = np.sqrt(1.5) * f_bands[b_idx - 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), btype='high', output='zpk')
        else:
            this_band = f_bands[b_idx] * np.array([1 / np.sqrt(1.5), np.sqrt(1.5)])
            z, p, k = butter(5, this_band / (fs // 2), btype='band', output='zpk')

        sos = zpk2sos(z, p, k)

        w, h = sosfreqz(sos, worN = len(input_signal) // 2 + 1)
        out_bands[:, b_idx] = np.fft.irfft(h)
        # somehow this does not work when the input signal is an impulse 
        # out_bands[:, b_idx] = sosfilt(sos, input_signal).squeeze()

    return out_bands

def biquad_to_tf(x, beta, alpha):
    x_pow = x ** np.array([0, -1, -2])
    
    band = 0    
    H = ((x_pow @ beta[band, :]) / (x_pow @ alpha[band, :]))
    for band in range(1, beta.shape[0]):
        H *= ((x_pow @ beta[band, :]) / (x_pow @ alpha[band, :]))
    
    return H

def rt2gain(rt, fs, delay=1):
    ''' convert rt (in seconds) to linear gain. If delay is not 1 it computes the 
    gain per sample '''
    Gdb = -60/fs/rt*delay
    return db2mag(Gdb)

def gain2rt(g, fs, delay=1):
    ''' convert linear gain to rt60 (in seconds) '''  
    return -3*delay/fs/np.log10(g)




def find_onset(rir, fs):
    win_len = int(fs * 0.002)

    overlap = 0.75
    energy_threshold = 0.01
    win = np.hanning(win_len)

    rir = np.pad(rir, (int(win_len * overlap), int(win_len * overlap)))
    hop = (1 - overlap)

    n_wins = np.floor(rir.shape[0] / (win_len * hop )- 1/2/hop ) 

    local_energy = []
    for i in range(1,int(n_wins - 1)):
        local_energy.append(
                np.sum( 
                    (rir[(i-1)*int(win_len*hop):(i-1)*int(win_len*hop) + win_len] ** 2) * win)
            )
    
    # discard trailing points 
    # remove (1/2/hop) to avoid map to negative time (center of window) 
    n_win_discard = (overlap/hop) - (1/2/hop) 

    local_energy = np.array(local_energy[int(n_win_discard):])

   
    loc = np.argwhere(local_energy > energy_threshold)[0][0]
    return int(win_len * hop * loc)    # one hopsize as safety margin 
