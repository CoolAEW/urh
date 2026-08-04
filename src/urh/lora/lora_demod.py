"""LoRa demodulator: symbol-level dechirp/FFT detection, preamble/SFD frame
sync on a continuous capture with unknown start offset, and full payload
recovery (deinterleave, Hamming-decode, dewhiten)."""

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


class LoRaSyncError(Exception):
    """Raised when no valid preamble/SFD could be located in the capture."""


def demod_symbol(window, sf, bw, fs):
    """Dechirp one symbol-length window and return the peak FFT bin (0..2**sf-1)."""
    n = 1 << sf
    down = chirp.reference_downchirp(sf, bw, fs)
    y = window * down
    mag = np.abs(np.fft.fft(y))[:n]
    return int(np.argmax(mag))


def _symbol_peakiness(window, sf, bw, fs, downchirp_ref=None):
    """peak-bin-magnitude / median-bin-magnitude. Peak-to-*median* (rather
    than peak-to-sum) is roughly SF-independent: sum-of-all-bins grows with
    N=2**sf even though the noise floor per bin doesn't, so peak/sum silently
    gets harder to clear at higher SF for no real detection-quality reason."""
    n = 1 << sf
    down = downchirp_ref if downchirp_ref is not None else chirp.reference_downchirp(sf, bw, fs)
    y = window * down
    mag = np.abs(np.fft.fft(y))[:n]
    peak_bin = int(np.argmax(mag))
    median = float(np.median(mag))
    peakiness = float(mag[peak_bin]) / median if median > 0 else 0.0
    return peak_bin, peakiness


def _matched_filter_magnitudes(iq, template):
    """|correlation| of `iq` against `template` at every valid lag, computed
    in one FFT pass (O(L log L)) rather than a per-lag loop. corr[lag] =
    sum_i iq[lag+i] * conj(template[i]), for lag in [0, len(iq)-len(template)].
    """
    L = len(iq)
    m = len(template)
    h = np.zeros(L, dtype=np.complex128)
    h[:m] = template
    # circular correlation of iq with zero-padded template; valid (non-wrapped)
    # for lag in [0, L-m], which is exactly the range we need.
    corr = np.fft.ifft(np.fft.fft(iq) * np.conj(np.fft.fft(h)))
    valid_len = L - m + 1
    return np.abs(corr[:valid_len])


def find_frame_start(
    iq,
    sf,
    bw,
    fs,
    n_preamble=DEFAULT_N_PREAMBLE,
    threshold_ratio=0.5,
):
    """Locate the LoRa preamble in a continuous capture with unknown start
    offset (leading/trailing noise, arbitrary alignment) and return
    (preamble_start, header_start) sample indices.

    Uses a single FFT-based matched filter against the known preamble
    upchirp (the optimal detector for a known signal in AWGN) instead of a
    per-sample brute-force search -- O(L log L) once instead of O(n_sym) FFTs
    per candidate offset, which made large SF/oversampling combinations
    impractically slow. Since every preamble symbol is an identical up0
    chirp, the matched filter produces a comb of near-equal peaks spaced
    exactly n_sym apart across the whole preamble; we take the first strong
    peak as the frame start.
    """
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    if len(iq) < (n_preamble + 2) * n_sym:
        raise LoRaSyncError("capture too short to contain a preamble")

    up0 = chirp.reference_upchirp(sf, bw, fs)
    mag = _matched_filter_magnitudes(iq, up0)

    peak_val = float(mag.max()) if len(mag) else 0.0
    if peak_val <= 0:
        raise LoRaSyncError("no signal energy found")

    # A relative threshold alone can never say "no signal" -- there's always
    # *some* max, even in pure noise. Require the peak to also clear the
    # noise floor by a healthy margin (peak-to-median, same rationale as
    # _symbol_peakiness: SF/OSR-independent, unlike peak-to-sum).
    noise_floor = float(np.median(mag))
    if peak_val < 6.0 * max(noise_floor, 1e-12):
        raise LoRaSyncError("no preamble-strength signal found (peak does not clear noise floor)")

    strong = np.where(mag > threshold_ratio * peak_val)[0]
    if len(strong) == 0:
        raise LoRaSyncError("no preamble-strength correlation peak found")

    # snap to the true local peak near the first strong sample (matched-filter
    # response is sharp but not necessarily a single sample wide)
    candidate = int(strong[0])
    lo, hi = max(0, candidate - 2), min(len(mag), candidate + 3)
    candidate = lo + int(np.argmax(mag[lo:hi]))

    # walk backward in whole-symbol steps while a strong peak persists, to
    # land on the *first* preamble symbol rather than some symbol within it
    cur = candidate
    while cur - n_sym >= 0 and mag[cur - n_sym] > threshold_ratio * peak_val:
        cur -= n_sym
    preamble_start = cur

    header_start = preamble_start + (n_preamble + 2) * n_sym + int(round(SFD_SYMBOLS * n_sym))
    return preamble_start, header_start


def _demod_symbols(iq, offset, count, sf, bw, fs):
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    symbols = []
    for i in range(count):
        start = offset + i * n_sym
        window = iq[start:start + n_sym]
        symbols.append(demod_symbol(window, sf, bw, fs))
    return symbols, offset + count * n_sym


def _decode_block_stream(raw_symbols, ppm, rdd, cr, shift_bits=0):
    """Gray-demap + deinterleave + Hamming-decode a whole stream of symbols
    (must be a multiple of `rdd` long) back into a nibble list."""
    assert len(raw_symbols) % rdd == 0
    nibbles = []
    uncorrectable_count = 0
    for block_start in range(0, len(raw_symbols), rdd):
        block_syms = raw_symbols[block_start:block_start + rdd]
        raw_vals = [phy.gray_decode(s >> shift_bits) for s in block_syms]
        codewords = phy.diagonal_deinterleave(raw_vals, ppm, rdd)
        for cw in codewords:
            nib, err = phy.hamming_decode_nibble(cw, cr)
            nibbles.append(nib)
            uncorrectable_count += int(err)
    return nibbles, uncorrectable_count


def decode_frame(
    iq,
    sf,
    bw,
    fs=None,
    n_preamble=DEFAULT_N_PREAMBLE,
    sync_word=DEFAULT_SYNC_WORD,
):
    """Full receive chain: locate frame, decode header, decode payload.

    Returns a dict: {payload: bytes, cr: int, uncorrectable_errors: int,
    sync_ok: bool, header_start: int}.
    """
    if fs is None:
        fs = bw

    preamble_start, header_start = find_frame_start(iq, sf, bw, fs, n_preamble)

    # sanity-check the sync word symbols (informational, not fatal)
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    sync_offset = preamble_start + n_preamble * n_sym
    sync_syms, _ = _demod_symbols(iq, sync_offset, 2, sf, bw, fs)
    expected_sync = [((sync_word >> 4) & 0xF) * 8, (sync_word & 0xF) * 8]
    sync_ok = sync_syms == expected_sync

    # header: sf-2 effective bits, always CR 4/8
    ppm_hdr = sf - 2
    rdd_hdr = 4 + HEADER_CR
    header_syms, payload_start = _demod_symbols(iq, header_start, rdd_hdr, sf, bw, fs)
    header_nibbles, header_errs = _decode_block_stream(
        header_syms, ppm_hdr, rdd_hdr, HEADER_CR, shift_bits=HEADER_SHIFT_BITS
    )
    payload_len = ((header_nibbles[0] & 0xF) << 4) | (header_nibbles[1] & 0xF)
    cr = header_nibbles[2] & 0xF
    if not (1 <= cr <= 4):
        raise LoRaSyncError(f"decoded implausible coding rate {cr} from header; sync likely wrong")

    # payload: full rate (sf), coding rate `cr`
    rdd = 4 + cr
    n_payload_nibbles = payload_len * 2
    n_blocks = (n_payload_nibbles + sf - 1) // sf
    n_payload_symbols = n_blocks * rdd

    payload_syms, _ = _demod_symbols(iq, payload_start, n_payload_symbols, sf, bw, fs)
    payload_nibbles, payload_errs = _decode_block_stream(payload_syms, sf, rdd, cr, shift_bits=0)
    payload_nibbles = payload_nibbles[:n_payload_nibbles]

    whitened_bytes = bytes(
        (payload_nibbles[i] << 4) | payload_nibbles[i + 1]
        for i in range(0, len(payload_nibbles), 2)
    )
    payload = phy.dewhiten(whitened_bytes)

    return {
        "payload": payload,
        "cr": cr,
        "uncorrectable_errors": header_errs + payload_errs,
        "sync_ok": sync_ok,
        "preamble_start": preamble_start,
        "header_start": header_start,
    }
