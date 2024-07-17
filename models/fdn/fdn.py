import numpy as np
from models.fdn.utility import *
from models.fdn.decayfit import *
from models.fdn.filters import *
from models.model_utils import find_direct_location, get_direct_window

from scipy.signal import firwin2
from scipy.linalg import qr

def householder_matrix(N):
    ''' 
    get NxN householder matrix from random vector of length N 
    '''
    u = np.random.rand(N,1)
    u = u / np.sqrt(np.sum(np.power(u, 2))) 
    return (np.identity(N) - 2*u*np.transpose(u)).astype('float32')

def orthogonal_matrix(N):
    ''' 
    get NxN random orthogonal matrix 
    '''
    # Generate a random n x n orthogonal real matrix
    A = np.random.randn(N, N)
    Q, _ = qr(A)

    # Ensure the determinant is positive to make it truly random orthogonal
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q

def polynomial_matrix(A, x):
    '''construct diagonal polynomial matri of order M x from coefficents A 
    and variables x of order M. Used for attenuation filter
    args:
        A (array):  array of coefficents (N, M)
        x (array):  array of variables (L,)
    ''' 
    N = A.shape[1]   # number of delay lines 
    M = A.shape[0]   # filter's order
    L = x.shape[0]   

    x = np.expand_dims(x, axis=-1)**(-np.arange(M))
    A_exp = np.zeros((L, N, N), np.complex_)
    for i in range(M):
        A_exp =+ np.einsum('i,jk->ijk', x[:,i], np.diag(A[i, :]))
    return A_exp






def absorptionFilters(frequency, targetRT60, filter_order, delays, fs, fdn_level=np.array(0)):
    '''
    Generate delay-proportional absorption filters
    Generates an FIR filter with given filter order such that placed in a
    recursion loop with a given delay, a specified reverberation time is    
    achieved. Multiple channels filters can be generated at once. The first
    and last elements of frequency must equal 0 and fs/2, respectively.
    args:
        frequency: Frequency points for T60 definition in [hertz]
        targetRT60: Target T60 in [seconds] and size [frequency, channels]
        filter_order: FIR filter order [samples]
        delays: Loop delay length in [samples] and size [1, channels]
        fs: Sampling frequency in [hertz]
    Adapted from Sebastian Schelcht' fdnToolbox
    '''
    # Abbreviation
    n_delays = len(delays)
    unit_freq = hertz2unit(frequency, fs)  

    # Compute filters per channel
    FIR = np.zeros((n_delays, filter_order + 1))

    if filter_order == 0:
        rt60 = targetRT60[0]
        delay = delays
        db = delay * rt2slope(rt60, fs)  
        FIR = db2mag(db)  
    else:
        for ch in range(n_delays):
            rt60 = targetRT60[:, ch]
            delay = delays[ch]
            delay = delay + np.ceil(filter_order / 2)
            db = delay * rt2slope(rt60, fs)  
            target_amp = db2mag(db)  
            target_amp += np.squeeze(fdn_level)
            # Filter Approximation with the window method
            FIR[ch, :] = firwin2(filter_order+1, unit_freq, target_amp)

    return FIR
    

def generate(rir, fBands, fs) : 
    T, A, N, level = get_fdn_EDCparam(rir, fBands[1:-1], n_slopes=1, sr=fs)
    T = np.pad(T, ((1,1), (0, 0)), 'edge')
    T = np.multiply(T, np.array([[0.9, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.5]]).transpose(1, 0))

    avg_rt60 = np.mean(T[2:5])
    # BINAURALIZATION OF OMNIDIRECTIONAL ROOM IMPULSE RESPONSES - ALGORITHM AND TECHNICAL EVALUATION
    mixing_time = int(0.05 * fs) if avg_rt60 <0.5 else int(0.1 * fs)

    # ---- FDN PARAMETERS ---- #

    # get filter coefficents 
    m = np.random.randint(low = 500, high = 3500, size=16)
    # get absorption filter
    G = absorptionGEQ(T, m, fs) # SOS 
    G = G / np.reshape(G[:,:,:,3], (len(m), 1, len(fBands)+1, 1))   # a0 = 1

    # feedback matrix
    A = orthogonal_matrix(len(m)) + 1j*np.zeros((len(m), len(m)))
    # input and output gains
    B = np.random.uniform(low=-1.0, high=1.0, size=(len(m), 1)) + 1j*np.zeros((len(m), 1))  
    # B = np.ones((len(m), 1)) + 1j*np.zeros((len(m), 1)) 
    C = np.random.uniform(low=-1.0, high=1.0, size=(1, len(m))) + 1j*np.zeros((1, len(m)))   
    # C = np.ones((1, len(m))) +  1j*np.zeros((1, len(m)))
    # d = np.array(0) # direct gain
    # initial level filter, attenuate top and bottom bands
    target_level = mag2db(np.pad(level, ((1,1), (0, 0)), 'edge'))
    target_level = target_level.squeeze() - np.array([5, 0, 0, 0, 0, 0, 0, 0, 5, 30])
    Ceq_sos = designGEQ(target_level)

    # get the trasfer function using frequency sampling method 
    num = 2*fs
    x = get_frequency_samples(num)

    # output filters
    Ceq = biquad_to_tf(np.reshape(x, (num, 1)), Ceq_sos[:, 0:3], Ceq_sos[:, 3:6])
    C = np.matmul(Ceq.reshape(num, 1), C).reshape(num, 1, len(m))

    # absorption filter
    Gch = np.zeros((len(x), len(m))) + 1j*np.zeros((len(x), len(m)))
    for ch in range(len(m)):
        Gch[:, ch] = biquad_to_tf(np.reshape(x, (num, 1)), G[ch,:,:,0:3].squeeze(), G[ch,:,:,3:6].squeeze())

    D_diag = (np.expand_dims(np.array(x), axis=-1)**np.array(m))    

    # this is so ugly, but there no python version of torch.diag_embed
    D = np.zeros((num, len(m), len(m)), dtype=np.complex_)
    G = np.zeros((num, len(m), len(m)), dtype=np.complex_)
    for i in range(num):
        for j in range(len(m)):
            D[i, j, j] = D_diag[i, j]
            G[i, j, j] = Gch[i, j]

    # compute FDN response
    H = np.matmul(C, np.matmul(np.linalg.inv(D - np.matmul(G, A)), B)).squeeze()
    h = np.fft.irfft(H, norm='ortho')
    h = h[:len(rir)]

    h = h / np.linalg.norm(h)
    

    # COMBINE 

    original_direct_location = find_direct_location(rir)
    fdn_direct_location = find_onset(h, fs)
    
    # Shift FDN to match the original direct location 

    shift_amount = original_direct_location - fdn_direct_location
    synth_rir_shifted = np.zeros_like(rir)
    if shift_amount < 0 : 
        # need to shift left 
        print ("FDN is more delayed than original")
        synth_rir_shifted[:len(h) + shift_amount] = h[abs(shift_amount):]
    else :
        # need to shift right 
        print ("FDN is earlier than original")
        synth_rir_shifted[shift_amount:] = h[:-shift_amount]

    synth_rir_shifted_full_length = np.zeros_like(rir)
    synth_rir_shifted_full_length[:len(synth_rir_shifted)] = synth_rir_shifted
    
    # Replace direct
    result_rir = synth_rir_shifted_full_length

    # # Match energy level 
    # late_energy_original = np.sqrt(np.mean(rir[mixing_time:]**2))
    # late_energy_generated = np.sqrt(np.mean(result_rir[mixing_time:]**2))
    # result_rir = result_rir/late_energy_generated*late_energy_original

    original_direct_window, original_direct_length = get_direct_window(rir, fs)
    original_direct = original_direct_window * rir
    original_late = rir[original_direct_length:]
    # late_energy = np.linalg.norm(original_late)
    late_energy = np.sqrt(np.mean(rir[mixing_time:]**2))
    generated_energy = np.sqrt(np.mean(result_rir[mixing_time:] **2 ))

    fdn_direct_window, _ = get_direct_window(synth_rir_shifted_full_length, fs)
    fdn_direct = fdn_direct_window * result_rir 
    
    result_rir -= fdn_direct
    result_rir =  result_rir /generated_energy * late_energy
    
    result_rir += original_direct 

    result_rir = np.clip(result_rir, a_min=-1, a_max=1)

    return result_rir
