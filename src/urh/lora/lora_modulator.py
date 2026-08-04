"""Synthetic LoRa modulator -- builds a complete, valid LoRa IQ frame from
arbitrary payload bytes. Used as the test oracle for the demodulator: there
is no ambiguity about what a correct decode should produce, because we
generated the frame ourselves.
"""

import numpy as np

from urh.lora import lora_phy as phy
from urh.lora import lora_chirp as chirp
from urh.lora.lora_frame_format import (
    DEFAULT_SYNC_WORD,
    DEFAULT_N_PREAMBLE,
    SFD_SYMBOLS,
    HEADER_CR,
    HEADER_SHIFT_BITS,
)


def _bytes_to_nibbles(data):
    nibbles = []
    for b in data:
        nibbles.append((b >> 4) & 0xF)
        nibbles.append(b & 0xF)
    return nibbles


def _build_header_nibbles(payload_len, cr, ppm_hdr):
    nibbles = [(payload_len >> 4) & 0xF, payload_len & 0xF, cr & 0xF]
    while len(nibbles) < ppm_hdr:
        nibbles.append(0)
    return nibbles[:ppm_hdr]


def _nibbles_to_symbols(nibbles, ppm, rdd, cr, shift_bits=0):
    """Hamming-encode `ppm` nibbles at a time into a block, diagonal-interleave,
    Gray-encode, and return the list of chirp-shift values (one per symbol)."""
    symbols = []
    for block_start in range(0, len(nibbles), ppm):
        block = nibbles[block_start:block_start + ppm]
        while len(block) < ppm:
            block.append(0)
        codewords = [phy.hamming_encode_nibble(nib, cr) for nib in block]
        raw_syms = phy.diagonal_interleave(codewords, ppm, rdd)
        for raw in raw_syms:
            symbols.append(phy.gray_encode(raw) << shift_bits)
    return symbols


def build_frame(
    payload,
    sf,
    bw,
    cr=4,
    fs=None,
    n_preamble=DEFAULT_N_PREAMBLE,
    sync_word=DEFAULT_SYNC_WORD,
):
    """Build a full LoRa IQ frame (numpy complex128 array).

    payload: bytes
    sf: spreading factor 7-12
    bw: bandwidth in Hz
    cr: payload coding rate, 1..4 (=> 4/5..4/8)
    fs: sample rate in Hz (defaults to bw, i.e. no oversampling)
    """
    if fs is None:
        fs = bw
    if not (7 <= sf <= 12):
        raise ValueError("sf must be 7..12")
    if not (1 <= cr <= 4):
        raise ValueError("cr must be 1..4")
    if len(payload) > 255:
        raise ValueError("payload too long (max 255 bytes)")

    ppm_hdr = sf - 2
    rdd_hdr = 4 + HEADER_CR

    parts = []

    # Preamble: n_preamble base upchirps (symbol 0)
    up0 = chirp.chirp_symbol(0, sf, bw, fs, downchirp=False)
    for _ in range(n_preamble):
        parts.append(up0)

    # Sync word: 2 upchirps at shift = nibble * 8
    sync_hi = ((sync_word >> 4) & 0xF) * 8
    sync_lo = (sync_word & 0xF) * 8
    parts.append(chirp.chirp_symbol(sync_hi, sf, bw, fs, downchirp=False))
    parts.append(chirp.chirp_symbol(sync_lo, sf, bw, fs, downchirp=False))

    # SFD: 2.25 downchirp symbol periods
    down0 = chirp.chirp_symbol(0, sf, bw, fs, downchirp=True)
    n_samples_sym = len(down0)
    parts.append(down0)
    parts.append(down0)
    parts.append(down0[: int(round(SFD_SYMBOLS % 1 * n_samples_sym))])

    # Header: length + cr, reduced rate (sf-2), always CR 4/8
    header_nibbles = _build_header_nibbles(len(payload), cr, ppm_hdr)
    header_symbols = _nibbles_to_symbols(
        header_nibbles, ppm_hdr, rdd_hdr, HEADER_CR, shift_bits=HEADER_SHIFT_BITS
    )
    for s in header_symbols:
        parts.append(chirp.chirp_symbol(s, sf, bw, fs, downchirp=False))

    # Payload: whitened, full rate (sf), coding rate cr
    whitened = phy.whiten(bytes(payload))
    payload_nibbles = _bytes_to_nibbles(whitened)
    rdd = 4 + cr
    payload_symbols = _nibbles_to_symbols(payload_nibbles, sf, rdd, cr, shift_bits=0)
    for s in payload_symbols:
        parts.append(chirp.chirp_symbol(s, sf, bw, fs, downchirp=False))

    return np.concatenate(parts)
