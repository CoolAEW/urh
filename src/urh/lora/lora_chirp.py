"""
LoRa chirp (CSS) waveform generation, shared by the modulator and demodulator.

Symbol `s` (0 <= s < 2**sf) is the base up-chirp cyclically shifted by `s`
bins. Using continuous normalized time tau = k * (bw/fs) for sample index k,
the closed-form phase below produces that cyclic shift "for free" via
aliasing when sampled at rate fs -- no explicit modulo/wrap-around handling
is needed, and it is well-defined for any fs >= bw (not just integer
oversampling ratios).

Dechirping a received symbol-s upchirp window with the conjugate of the
symbol-0 upchirp (a "downchirp") multiplies out to a pure discrete tone at
FFT bin `s`:

    upchirp_s[k]   = exp(j*2*pi*(tau^2/(2N) + tau*(s/N - 0.5)))
    downchirp_0[k] = exp(-j*2*pi*(tau^2/(2N) - 0.5*tau))
    upchirp_s[k] * downchirp_0[k] = exp(j*2*pi*k*(bw/fs)*(s/N))

which is an exact-bin-s tone in a length-round(fs/bw*N) DFT.
"""

from functools import lru_cache

import numpy as np


def samples_per_symbol(sf, bw, fs):
    return int(round((fs / bw) * (1 << sf)))


def chirp_symbol(symbol, sf, bw, fs, downchirp=False):
    """Generate one LoRa symbol's IQ waveform.

    symbol: integer in [0, 2**sf)
    sf: spreading factor (7-12)
    bw: LoRa bandwidth in Hz
    fs: sample rate in Hz (>= bw)
    downchirp: if True, generate the downchirp (used for SFD / dechirping)
    """
    n = 1 << sf
    n_samples = samples_per_symbol(sf, bw, fs)
    k = np.arange(n_samples)
    tau = k * (bw / fs)
    sign = -1.0 if downchirp else 1.0
    phase = 2 * np.pi * sign * ((tau ** 2) / (2 * n) + tau * (symbol / n - 0.5))
    return np.exp(1j * phase).astype(np.complex128)


@lru_cache(maxsize=None)
def reference_downchirp(sf, bw, fs):
    """Cached: (sf, bw, fs) is invariant across an entire scan (many chunks
    x many n_preamble/sync_word/CFO-iteration passes), and this array is
    never mutated by any caller (only ever multiplied against, e.g.
    `window * down`, which allocates a new array) -- safe to share.
    """
    return chirp_symbol(0, sf, bw, fs, downchirp=True)


@lru_cache(maxsize=None)
def reference_upchirp(sf, bw, fs):
    """Cached -- see reference_downchirp."""
    return chirp_symbol(0, sf, bw, fs, downchirp=False)
