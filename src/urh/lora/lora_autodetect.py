"""Blind SF/BW (and, opportunistically, preamble-length/sync-word)
auto-detection for a raw LoRa capture with unknown parameters.

Pure Python/numpy, no PyQt dependency -- mirrors
urh.ainterpretation.AutoInterpretation.estimate() being independent of
Signal/Qt, so it's usable standalone (tests, scripts) as well as from
LoRaDecoderDialog's "Auto-detect" button.

Two-stage cost control:
  1. Cheap: score every (SF, BW) combination in the standard grid (6 SF x
     10 BW = 60 combos) using preamble_match_score's peak-to-noise-floor
     ratio -- the same quantity find_frame_start already computes
     internally and, until now, just thresholded and discarded. No FEC/
     payload decode is attempted at this stage.
  2. Expensive: run a real decode_frame attempt (via scan_chunk_for_frames,
     trying a small set of n_preamble/sync_word candidates the same way
     manual scanning already does) only on the top-K cheap-stage winners,
     to confirm/select the final answer and opportunistically recover
     n_preamble/sync_word too.
"""

from dataclasses import dataclass

from urh.lora.lora_demod import (
    LoRaSyncError,
    NOISE_FLOOR_CLEARANCE,
    preamble_match_score,
    scan_chunk_for_frames,
)
from urh.lora.lora_frame_format import (
    DEFAULT_N_PREAMBLE,
    DEFAULT_SYNC_WORD,
    STANDARD_BANDWIDTHS,
    STANDARD_SPREADING_FACTORS,
)

DEFAULT_TOP_K = 3
#: Secondary preamble-length/sync-word candidates tried in the expensive
#: stage, mirroring what manual scanning already tries by default.
DEFAULT_N_PREAMBLE_OPTIONS = (DEFAULT_N_PREAMBLE, 16)
DEFAULT_SYNC_WORD_OPTIONS = (DEFAULT_SYNC_WORD, 0x12)
#: Target oversampling ratio after decimating a candidate BW down from a
#: high native sample rate for the cheap ranking pass -- comfortably above
#: 1 (chirp generation requires fs >= bw) without being wastefully high.
_CHEAP_STAGE_TARGET_OSR = 4


@dataclass
class AutoDetectCandidate:
    sf: int
    bw: float
    score: float  # peak-to-noise-floor ratio from the cheap stage
    decode_result: dict = None  # populated for top-K candidates by the expensive stage
    n_preamble: int = None
    sync_word: int = None


def _decimate(iq, factor):
    """Cheap boxcar-average decimation (no scipy available in this
    environment). Acts as a crude low-pass/anti-alias step before
    downsampling. Precision only needs to be good enough for *relative*
    ranking in the cheap stage -- the expensive stage's real decode attempt
    on the winning candidate(s) runs against the un-decimated snippet and
    would reject any false positive decimation artifacts introduced."""
    if factor <= 1:
        return iq
    n = (len(iq) // factor) * factor
    if n == 0:
        return iq[:0]
    return iq[:n].reshape(-1, factor).mean(axis=1)


def _cheap_score(iq_snippet, fs, sf, bw, n_preamble):
    """One (SF, BW) candidate's cheap-stage score, or None if this
    candidate doesn't apply (snippet too short at this SF/BW, BW exceeds
    what the native sample rate can represent, or no signal energy at
    all)."""
    if bw > fs:
        return None  # physically impossible to represent at this sample rate

    decimation = max(1, int(fs / (bw * _CHEAP_STAGE_TARGET_OSR)))
    decimated_iq = _decimate(iq_snippet, decimation)
    decimated_fs = fs / decimation
    if decimated_fs < bw:
        return None

    try:
        _, peak_val, noise_floor = preamble_match_score(
            decimated_iq, sf, bw, decimated_fs, n_preamble=n_preamble
        )
    except LoRaSyncError:
        return None  # snippet too short for this candidate, or zero energy

    return peak_val / max(noise_floor, 1e-12)


def estimate(
    iq_snippet,
    fs,
    sf_candidates=STANDARD_SPREADING_FACTORS,
    bw_candidates=None,
    top_k=DEFAULT_TOP_K,
    n_preamble_options=DEFAULT_N_PREAMBLE_OPTIONS,
    sync_word_options=DEFAULT_SYNC_WORD_OPTIONS,
):
    """Blind search for the (SF, BW) combination that best explains
    iq_snippet. Returns a list of AutoDetectCandidate, best (highest cheap
    score) first; only the first `top_k` have `decode_result`/`n_preamble`/
    `sync_word` populated (from the expensive stage), the rest carry just
    `score` for visibility into what else was considered.

    A short snippet (not the whole capture) is expected -- the point is to
    identify parameters quickly, not to do the full scan at every
    candidate. Real captures/threshold defaults here are the same kind of
    "reasonable starting point, needs real-capture tuning" caveat CFO
    correction shipped with earlier.
    """
    if bw_candidates is None:
        bw_candidates = [bw for _, bw in STANDARD_BANDWIDTHS]

    candidates = []
    for sf in sf_candidates:
        for bw in bw_candidates:
            score = _cheap_score(iq_snippet, fs, sf, bw, n_preamble_options[0])
            if score is None or score < NOISE_FLOOR_CLEARANCE:
                continue
            candidates.append(AutoDetectCandidate(sf=sf, bw=bw, score=score))

    candidates.sort(key=lambda c: -c.score)

    for candidate in candidates[:top_k]:
        hits = scan_chunk_for_frames(
            iq_snippet, candidate.sf, candidate.bw, fs, n_preamble_options, sync_word_options
        )
        if not hits:
            continue
        hits.sort(key=lambda h: -h["confidence"])
        best = hits[0]
        candidate.decode_result = best
        candidate.n_preamble = best["n_preamble"]
        candidate.sync_word = best["sync_word"]

    return candidates
