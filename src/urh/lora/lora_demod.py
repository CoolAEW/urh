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


def _dechirp_fft(window, sf, bw, fs, downchirp_ref=None):
    """Dechirp one symbol-length window against the reference downchirp and
    return the first 2**sf complex FFT bins. Magnitude-argmax over this is
    the demodulated symbol value; the complex value at a given bin (not just
    its magnitude) is also used for fine CFO phase estimation."""
    n = 1 << sf
    down = downchirp_ref if downchirp_ref is not None else chirp.reference_downchirp(sf, bw, fs)
    y = window * down
    return np.fft.fft(y)[:n]


def demod_symbol(window, sf, bw, fs):
    """Dechirp one symbol-length window and return the peak FFT bin (0..2**sf-1)."""
    mag = np.abs(_dechirp_fft(window, sf, bw, fs))
    return int(np.argmax(mag))


def _demod_symbol_with_peakiness(window, sf, bw, fs, downchirp_ref):
    """Dechirp one symbol-length window and return (bin, peakiness) from a
    single FFT pass -- shared by demod_symbol/_symbol_peakiness below and by
    _demod_symbols' per-symbol loop, so getting peakiness "for free" during
    header/payload demod doesn't cost a second FFT per symbol.

    peakiness = peak-bin-magnitude / median-bin-magnitude. Peak-to-*median*
    (rather than peak-to-sum) is roughly SF-independent: sum-of-all-bins
    grows with N=2**sf even though the noise floor per bin doesn't, so
    peak/sum silently gets harder to clear at higher SF for no real
    detection-quality reason. peakiness is 0.0 for an exactly-zero/flat
    window (median==0), which is the real, observed signature of trailing
    silence in a live capture (see find_frame_start's collapse-detection
    callers) rather than an edge case to special-case away.
    """
    mag = np.abs(_dechirp_fft(window, sf, bw, fs, downchirp_ref=downchirp_ref))
    peak_bin = int(np.argmax(mag))
    peak_mag = float(mag[peak_bin])
    # Floor the median to a small fraction of the peak before dividing: with
    # near-zero real noise (idealized/noiseless synthetic signals, or very
    # high real SNR), the non-peak bins are dominated by float64 rounding
    # residue rather than anything physical, and an unclamped median swings
    # wildly (observed: peakiness varying ~300x between symbols of the
    # *same* clean synthetic frame purely from this effect), which makes
    # peakiness useless as a stable per-frame reference for collapse
    # detection. Flooring caps peakiness at a large-but-stable value in that
    # regime while leaving genuinely noisy/collapsed windows (where the
    # floor isn't the binding constraint) unaffected.
    median = max(float(np.median(mag)), peak_mag * 1e-6)
    peakiness = peak_mag / median if median > 0 else 0.0
    return peak_bin, peakiness


def _symbol_peakiness(window, sf, bw, fs, downchirp_ref=None):
    """Thin wrapper over _demod_symbol_with_peakiness for standalone
    (non-_demod_symbols-loop) callers that only want peakiness, not a full
    symbol-block decode."""
    down = downchirp_ref if downchirp_ref is not None else chirp.reference_downchirp(sf, bw, fs)
    return _demod_symbol_with_peakiness(window, sf, bw, fs, down)


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


def estimate_cfo_hz(iq, preamble_start, n_preamble, sf, bw, fs):
    """Estimate carrier frequency offset (CFO) from the preamble's repeated
    symbol-0 upchirps, in two stages:

    Coarse (integer bin): with a CFO, every preamble upchirp dechirps to the
    same nonzero bin B (mod 2**sf) instead of bin 0. Take the bin most
    preamble symbols agree on. B > 2**(sf-1) is treated as a negative offset
    (aliased the other way), the usual convention for a symmetric range.

    Fine (sub-bin): a coarse bin estimate alone leaves up to +/-0.5 bin of
    residual error, which spreads FFT energy across adjacent bins rather
    than concentrating it -- this shows up as a *phase rotation* between
    consecutive identical preamble symbols at the coarse bin. Average that
    phase difference and convert to Hz. This step is only reliable once
    coarse correction has narrowed things to within one bin: phase-only
    estimation between two symbols is inherently ambiguous (aliased) for
    offsets larger than +/-0.5 cycle per symbol period.

    A real CFO also shifts *where in time* a chirp matched filter's peak
    lands (the chirp-radar "range-Doppler coupling" effect), so the given
    preamble_start -- typically from find_frame_start on the *uncorrected*
    signal -- can be off by a handful of samples whenever a real CFO is
    present, which in turn can make a single estimate pass here inaccurate.
    This function does not compensate for that itself (an ad-hoc local
    search over nearby sample offsets turned out to overfit noise and was
    worse than not searching at all); callers wanting an accurate estimate
    under a potentially-large CFO should call this, correct, re-locate the
    frame with find_frame_start on the corrected signal, and repeat until
    the estimate stops changing much -- see decode_frame's correct_cfo
    handling, which iterates for exactly this reason.

    Returns the estimated offset in Hz (positive = signal appears shifted
    up in frequency; caller should derotate by -f_est to correct it).
    """
    n = 1 << sf
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    down = chirp.reference_downchirp(sf, bw, fs)

    fft_per_symbol = []
    for i in range(n_preamble):
        start = preamble_start + i * n_sym
        window = iq[start : start + n_sym]
        if len(window) < n_sym:
            break
        fft_per_symbol.append(_dechirp_fft(window, sf, bw, fs, downchirp_ref=down))

    if not fft_per_symbol:
        return 0.0

    bins = [int(np.argmax(np.abs(fft_vals))) for fft_vals in fft_per_symbol]
    values, counts = np.unique(bins, return_counts=True)
    b_coarse = int(values[np.argmax(counts)])
    b_signed = b_coarse if b_coarse <= n // 2 else b_coarse - n
    bin_width_hz = bw / n
    coarse_hz = b_signed * bin_width_hz

    vals_at_coarse_bin = [fft_vals[b_coarse] for fft_vals in fft_per_symbol]
    sym_period = n_sym / fs
    phase_diffs = [
        np.angle(np.conj(vals_at_coarse_bin[i]) * vals_at_coarse_bin[i + 1])
        for i in range(len(vals_at_coarse_bin) - 1)
    ]
    fine_hz = (float(np.mean(phase_diffs)) / (2 * np.pi) / sym_period) if phase_diffs else 0.0

    return coarse_hz + fine_hz


def apply_cfo_correction(iq, f_offset_hz, fs, start=0):
    """Derotate iq[start:] by -f_offset_hz (a no-op copy if f_offset_hz==0),
    using each sample's true absolute index for the time base so phase stays
    continuous across the correction boundary. Returns a new array; leaves
    the original untouched."""
    if f_offset_hz == 0.0:
        return iq
    n = np.arange(start, len(iq))
    rotation = np.exp(-1j * 2 * np.pi * f_offset_hz * n / fs).astype(iq.dtype)
    return np.concatenate([iq[:start], iq[start:] * rotation])


def _demod_symbols(iq, offset, count, sf, bw, fs):
    """Demod `count` consecutive symbols starting at `offset`. Returns
    (symbols, peakiness_list, new_offset) -- peakiness rides along for free
    since it's derived from the same per-symbol FFT the bin value already
    needed (see _demod_symbol_with_peakiness), letting callers detect a
    mid-frame collapse into trailing noise without a second demod pass."""
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    if offset + count * n_sym > len(iq):
        raise LoRaSyncError("not enough samples remaining for requested symbol block")
    down = chirp.reference_downchirp(sf, bw, fs)
    symbols = []
    peakiness = []
    for i in range(count):
        start = offset + i * n_sym
        window = iq[start:start + n_sym]
        b, p = _demod_symbol_with_peakiness(window, sf, bw, fs, down)
        symbols.append(b)
        peakiness.append(p)
    return symbols, peakiness, offset + count * n_sym


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


# --- mid-frame collapse detection (see decode_frame) ---
# A payload symbol is "collapsed" (no real signal) if its peakiness drops
# below this fraction of the frame's own header-derived reference peakiness.
# Not derivable from synthetic AWGN alone -- real trailing silence dechirps
# to a stable near-zero peakiness in a way generic noise doesn't reproduce;
# starting value, expected to need retuning against real captures.
PAYLOAD_COLLAPSE_FRACTION = 0.3
# Require this many *consecutive* collapsed symbols before truncating, so a
# momentary real-RF fade/null doesn't kill an otherwise-good frame.
PAYLOAD_COLLAPSE_RUN_LENGTH = 4

# --- composite confidence score weights (see decode_frame) ---
# Multiplicative penalty when sync_ok is False -- not a hard gate, since a
# sync-word mismatch with otherwise-clean FEC has already been a genuinely
# promising, worth-investigating signal in real captures (see LORA_PLAN.md).
CONFIDENCE_SYNC_FAIL_PENALTY = 0.3
# error rate -> exp(-rate * K): K chosen so ~1% errors scores ~0.92 and ~50%
# (noise-level) scores ~0.02 -- a soft curve, not a hard cutoff.
CONFIDENCE_ERROR_RATE_K = 8.0
CONFIDENCE_WEIGHT_ERROR_RATE = 0.4
# The strongest, most physically-grounded signal: whether the payload's
# symbols stayed as peaky as the header's throughout, i.e. real signal
# rather than a collapse into trailing noise.
CONFIDENCE_WEIGHT_PEAKINESS = 0.4
# Weighted lower than peakiness: legitimate payloads can have genuinely
# low-entropy stretches (zero-padding, repeated fields), so this is a soft
# secondary signal, not a gate.
CONFIDENCE_WEIGHT_ENTROPY = 0.2
# Below this many payload symbols, entropy is too noisy a statistic to be
# meaningful -- treat as neutral (1.0) rather than penalizing short payloads.
CONFIDENCE_ENTROPY_MIN_SYMBOLS = 8


def _collapse_run_start(peakiness_list, reference, fraction, run_length):
    """Index of the first symbol in a *sustained* run of >= run_length
    consecutive symbols with peakiness < fraction*reference, or None if no
    such run exists. reference<=0 (a degenerate/silent header -- shouldn't
    normally happen since find_frame_start already gates on signal presence,
    but guard anyway) disables collapse detection rather than flagging
    everything."""
    if reference <= 0:
        return None
    threshold = fraction * reference
    run_start = None
    run_len = 0
    for i, p in enumerate(peakiness_list):
        if p < threshold:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= run_length:
                return run_start
        else:
            run_len = 0
    return None


def _symbol_entropy_score(symbols, sf):
    """Shannon entropy of the demodulated-bin histogram, normalized to
    [0, 1] against the maximum possible (log2(2**sf), i.e. uniform bin
    usage). A degenerate constant-symbol run (the observed real-world
    trailing-noise failure mode) scores 0.0 here. Short payloads don't carry
    enough symbols for entropy to mean much -- see
    CONFIDENCE_ENTROPY_MIN_SYMBOLS."""
    if len(symbols) < CONFIDENCE_ENTROPY_MIN_SYMBOLS:
        return 1.0
    _, counts = np.unique(symbols, return_counts=True)
    probs = counts / len(symbols)
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = sf  # log2(2**sf)
    return entropy / max_entropy if max_entropy > 0 else 1.0


def decode_frame(
    iq,
    sf,
    bw,
    fs=None,
    n_preamble=DEFAULT_N_PREAMBLE,
    sync_word=DEFAULT_SYNC_WORD,
    correct_cfo=True,
):
    """Full receive chain: locate frame, decode header, decode payload.

    Returns a dict: {payload: bytes, cr: int, uncorrectable_errors: int,
    sync_ok: bool, header_start: int, cfo_hz: float, confidence: float,
    payload_truncated: bool, truncated_at_symbol: int or None}.

    If real signal ends partway through the payload (trailing noise/silence
    after a genuinely short transmission), that trailing region is detected
    via a sustained drop in per-symbol peakiness relative to the header and
    the payload is *truncated* to what decoded before the collapse rather
    than either trusting clearly-bogus trailing bytes or discarding an
    otherwise-correct partial decode outright -- see PAYLOAD_COLLAPSE_*.
    """
    if fs is None:
        fs = bw

    preamble_start, header_start = find_frame_start(iq, sf, bw, fs, n_preamble)

    cfo_hz = 0.0
    if correct_cfo:
        # Iterate estimate -> correct -> re-locate rather than a single
        # pass: a CFO couples into the matched filter's peak *time* for a
        # chirp signal (the chirp-radar "range-Doppler coupling" effect --
        # an uncorrected offset shifts where the correlation peak lands),
        # so preamble_start computed on the *uncorrected* signal can be off
        # by a handful of samples, which in turn makes the first CFO
        # estimate inaccurate. Each correction pass removes most of the
        # remaining offset, which sharpens the next re-located
        # preamble_start, which sharpens the next estimate -- converges in
        # a couple of rounds rather than needing to get it exactly right in
        # one shot.
        for _ in range(3):
            step_hz = estimate_cfo_hz(iq, preamble_start, n_preamble, sf, bw, fs)
            if step_hz == 0.0:
                break
            iq = apply_cfo_correction(iq, step_hz, fs, start=0)
            cfo_hz += step_hz
            preamble_start, header_start = find_frame_start(iq, sf, bw, fs, n_preamble)

    # sanity-check the sync word symbols (informational, not fatal)
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    sync_offset = preamble_start + n_preamble * n_sym
    sync_syms, _, _ = _demod_symbols(iq, sync_offset, 2, sf, bw, fs)
    # See lora_modulator.build_frame: nibble shift is 2**(sf-4), not a fixed
    # *8 (only correct at SF=7).
    sync_shift = 1 << (sf - 4)
    expected_sync = [((sync_word >> 4) & 0xF) * sync_shift, (sync_word & 0xF) * sync_shift]
    sync_ok = sync_syms == expected_sync

    # header: sf-2 effective bits, always CR 4/8
    ppm_hdr = sf - 2
    rdd_hdr = 4 + HEADER_CR
    header_syms, header_peakiness, payload_start = _demod_symbols(
        iq, header_start, rdd_hdr, sf, bw, fs
    )
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

    payload_syms, payload_peakiness, _ = _demod_symbols(
        iq, payload_start, n_payload_symbols, sf, bw, fs
    )
    payload_nibbles, payload_errs = _decode_block_stream(payload_syms, sf, rdd, cr, shift_bits=0)
    payload_nibbles = payload_nibbles[:n_payload_nibbles]

    # Mid-frame collapse: use the header's own peakiness as this frame's
    # reference level (SNR varies across captures, so a fixed global
    # constant would be fragile) and look for a sustained drop in the
    # payload relative to it.
    header_ref_peakiness = float(np.median(header_peakiness)) if header_peakiness else 0.0
    collapse_symbol = _collapse_run_start(
        payload_peakiness, header_ref_peakiness,
        PAYLOAD_COLLAPSE_FRACTION, PAYLOAD_COLLAPSE_RUN_LENGTH,
    )
    payload_truncated = collapse_symbol is not None
    if payload_truncated:
        good_blocks = collapse_symbol // rdd  # only fully-good blocks are trustworthy
        good_nibbles = good_blocks * sf
        payload_nibbles = payload_nibbles[: min(len(payload_nibbles), good_nibbles)]
    # dewhiten operates byte-wise; drop a dangling odd nibble if truncation
    # landed mid-byte.
    if len(payload_nibbles) % 2:
        payload_nibbles = payload_nibbles[:-1]

    whitened_bytes = bytes(
        (payload_nibbles[i] << 4) | payload_nibbles[i + 1]
        for i in range(0, len(payload_nibbles), 2)
    )
    payload = phy.dewhiten(whitened_bytes)

    uncorrectable_errors = header_errs + payload_errs
    total_nibbles = len(header_nibbles) + n_payload_nibbles
    error_rate = uncorrectable_errors / total_nibbles if total_nibbles else 0.0
    error_score = float(np.exp(-error_rate * CONFIDENCE_ERROR_RATE_K))

    if payload_peakiness:
        good_count = collapse_symbol if payload_truncated else len(payload_peakiness)
        peakiness_score = good_count / len(payload_peakiness)
    else:
        peakiness_score = 1.0  # zero-length payload: nothing to collapse

    entropy_score = _symbol_entropy_score(payload_syms, sf) if payload_syms else 1.0

    confidence = (
        CONFIDENCE_WEIGHT_ERROR_RATE * error_score
        + CONFIDENCE_WEIGHT_PEAKINESS * peakiness_score
        + CONFIDENCE_WEIGHT_ENTROPY * entropy_score
    )
    if not sync_ok:
        confidence *= CONFIDENCE_SYNC_FAIL_PENALTY
    confidence = max(0.0, min(1.0, confidence))

    return {
        "payload": payload,
        "cr": cr,
        "uncorrectable_errors": uncorrectable_errors,
        "sync_ok": sync_ok,
        "preamble_start": preamble_start,
        "header_start": header_start,
        "cfo_hz": cfo_hz,
        "confidence": confidence,
        "payload_truncated": payload_truncated,
        "truncated_at_symbol": collapse_symbol,
    }


def scan_chunk_for_frames(iq_chunk, sf, bw, fs, n_preamble_options, sync_word_options):
    """Try every (n_preamble, sync_word) combination against one chunk of IQ
    samples. Returns a list of decode_frame result dicts for every attempt
    that found *a* frame-sync trigger (not necessarily sync_ok=True -- callers
    decide what to trust), each annotated with the n_preamble/sync_word used.
    """
    hits = []
    for n_preamble in n_preamble_options:
        for sync_word in sync_word_options:
            try:
                result = decode_frame(
                    iq_chunk, sf, bw, fs=fs, n_preamble=n_preamble, sync_word=sync_word
                )
            except LoRaSyncError:
                continue
            result["n_preamble"] = n_preamble
            result["sync_word"] = sync_word
            hits.append(result)
    return hits


def scan_for_frames(
    iq,
    sf,
    bw,
    fs,
    n_preamble_options=(DEFAULT_N_PREAMBLE,),
    sync_word_options=(DEFAULT_SYNC_WORD,),
    chunk_sec=3.0,
    overlap_sec=0.5,
    progress_cb=None,
    should_stop=None,
):
    """Chunked scan of a (possibly long) in-memory IQ array for a LoRa frame,
    avoiding one huge FFT over the whole capture -- impractical at multi-
    minute captures, and gives no opportunity to report progress or cancel.

    Calls progress_cb(chunk_idx, total_chunks, chunk_offset_sec) between
    chunks if given. Calls should_stop() between chunks if given and stops
    early if it returns True.

    Returns (hits, max_mags): hits is a list of decode_frame result dicts
    (each also carrying chunk_idx/chunk_offset_samples/n_preamble/sync_word),
    max_mags is the per-chunk peak |iq| magnitude -- useful for telling "no
    signal present" apart from "signal present but wouldn't sync".
    """
    chunk_samples = max(1, int(chunk_sec * fs))
    overlap_samples = int(overlap_sec * fs)
    step = max(1, chunk_samples - overlap_samples)
    total_samples = len(iq)
    total_chunks = max(1, -(-total_samples // step))  # ceil div

    hits = []
    max_mags = []
    offset = 0
    chunk_idx = 0
    while offset < total_samples:
        if should_stop is not None and should_stop():
            break
        chunk = iq[offset : offset + chunk_samples]
        if len(chunk) == 0:
            break
        max_mags.append(float(np.max(np.abs(chunk))))

        for result in scan_chunk_for_frames(
            chunk, sf, bw, fs, n_preamble_options, sync_word_options
        ):
            result["chunk_idx"] = chunk_idx
            result["chunk_offset_samples"] = offset
            hits.append(result)

        if progress_cb is not None:
            progress_cb(chunk_idx, total_chunks, offset / fs)

        offset += step
        chunk_idx += 1

    return hits, max_mags
