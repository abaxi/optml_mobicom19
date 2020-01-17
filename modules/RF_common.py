import numpy as np
import random

def get_lambs(cf, bw, nfft):
    c = 3e8
    f = np.fft.fftfreq(nfft)
    f = f*bw
    cf = f + cf
    l1 = []
    for f in cf:
        l1.append(c/f)
    return np.array(l1)

def get_synth_params_sep(n_paths, lo, hi, sep=None):
    '''
    using random values for the 4-tuples for
    each path. "sep" is the minimum distance between components.
    '''
    if sep == None:
        return get_synth_params(n_paths, lo, hi)
    
    np.random.seed()
    div = 10000.0
    hi = hi*div
    lo = lo*div
    population = np.arange(lo, hi, 2*sep*div)
    assert len(population)>=n_paths, " assert error: get_synth_params_sep(...): population size smaller than n_paths"
    d_ns = np.array(random.sample(population,n_paths)) 
    fractional = np.random.rand(n_paths)*sep*div
    d_ns = d_ns+fractional
    d_ns = d_ns/div
    if n_paths >1:
        assert max(d_ns)-min(d_ns)>=sep, " assert error: get_synth_params_sep(...): min separation assertion failed"

    a_ns = np.random.rand(n_paths)
    a_ns = a_ns/np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    psi_ns = np.cos(np.random.rand(n_paths)*np.pi)         ### 0 to pi range

    return d_ns, a_ns, phi_ns, psi_ns

def get_synth_params(n_paths, lo, hi):
    '''
    using random values for the 4-tuples for
    each path
    '''
    np.random.seed()
    div = 10000.0
    hi = hi*div
    lo = lo*div
    d_ns = np.random.randint(lo,hi,n_paths).astype(float)/div ### to add a fractional part

    a_ns = np.random.rand(n_paths)
    a_ns = a_ns/np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    psi_ns = np.cos(np.random.rand(n_paths)*np.pi)         ### 0 to pi range

    return d_ns, a_ns, phi_ns, psi_ns

def get_chans_from_params(params, K, sep, lambs):
    '''
    given the params/4-tuples for the physical paths, compute the channel
    across K antenna for given wavelengths.
    Based on Equation 3 of paper
    '''
    d_ns, a_ns, phi_ns, psi_ns = params
    N = len(d_ns)
    I = len(lambs)
    H = np.zeros([I,K]).astype(np.complex)

    for i_wl in range(len(lambs)):
        wl = lambs[i_wl]
        ### based on eq 3
        for i_K in range(K):          ###for each antenna
            t = 0
            for i_N in range(N):      ###for each path
                c1 = a_ns[i_N]*np.exp((-2j*np.pi*d_ns[i_N]/wl) + 1j*phi_ns[i_N])
                c1 = c1*np.exp(-2j*np.pi*(i_K)*sep*psi_ns[i_N]/wl)
                H[i_wl, i_K] += c1
    return H

def add_noise_snr_range(chans, min_snr, max_snr):
    n = chans.shape[0]
    r,c = chans.shape
    snrs = np.random.randint(min_snr, max_snr+1, n)
    scale = np.power(10, -1*snrs/20.0)
    scale = scale.reshape([-1,1])
    noise = np.random.randn(r,c)+1j*np.random.randn(r,c)
    noise = noise/np.abs(noise)
    noise = scale*noise
    chans = chans+noise
    return chans