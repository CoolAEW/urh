# LoRa Decoding Support — Implementation Plan

## Why this is a separate feature, not a new `modulation_type`

URH's existing demod pipeline (`Signal.py` / `signal_functions.afp_demod`) is a
per-*sample* amplitude/frequency/phase threshold detector feeding a fixed set
of modulation types (ASK, FSK, PSK, GFSK/OQPSK), each producing a bitstream at
a fixed `samples_per_symbol`. LoRa (chirp spread spectrum) is fundamentally
per-*symbol*: each symbol is a whole chirp (2^SF samples at the symbol rate),
identified by dechirping (multiply by conjugate reference chirp) then taking
the FFT peak bin. It does not produce a raw bitstream that maps onto that
pipeline, and `plugins/` operate on already-demodulated bits (`ProtocolAnalyzer`
output), not raw IQ — so neither the modulation-type slot nor the plugin
system is the right hook.

**Decision:** build a standalone "LoRa Decoder" dialog
(`controller/dialogs/LoRaDecoderDialog.py`, pattern-matched on
`SpectrumDialogController.py` / `SignalDetailsDialog.py`) that takes a loaded
`Signal`'s raw IQ (`signal.data`) plus user-entered SF (7-12), BW (125/250/500
kHz), and coding rate, and runs a fully independent LoRa RX chain producing a
hex/byte payload dump. This is additive and never touches the existing demod
path, so zero regression risk to ASK/FSK/PSK/GFSK. A thin core module
(`urh/dev/native/lib` sibling, plain Python/numpy — no Cython needed at first,
FFT-per-symbol is cheap) holds the actual DSP so it's independently unit
testable without Qt.

## Algorithm (confirmed via websearch against gr-lora/rpp0 and academic
reverse-engineering papers — matches training-data understanding)

TX chain: whiten → Hamming-encode → interleave → Gray-encode → chirp-modulate.
RX chain (reverse): dechirp → FFT → argmax → Gray-demap → deinterleave →
Hamming-decode → dewhiten.

- **Chirp modulation**: base upchirp spans BW over `2^SF / BW` seconds;
  symbol `s` is the base chirp cyclically shifted by `s` bins.
- **Dechirp/demod**: multiply received symbol window by conjugate *downchirp*,
  FFT, symbol = index of magnitude peak (non-coherent detection).
- **Preamble/sync**: repeated identical upchirps (autocorrelation between
  consecutive symbols > ~0.9) → sync word → two-and-a-quarter downchirps
  (SFD) mark frame start; the quarter-symbol offset is the reference point
  for header timing.
- **Header** (explicit mode): first block uses SF-2 effective rate, always
  CR 4/8 Hamming, and carries payload length + coding rate for the rest of
  the frame.
- **Interleaving**: `I(i, j) = D(j, (i - j - 1) mod SF)`.
- **Hamming**: (4, n) codes, n ∈ {8,7,6} depending on CR 4/8..4/5 (4/6 is
  detect-only, no correction).
- **Whitening**: XOR payload against a known PRBS byte sequence (LFSR
  `x^8+x^6+x^5+x^4+1`); safer to ship the precomputed reference whitening
  table (as gr-lora and the Semtech reference code do) rather than compute
  the LFSR live, to avoid subtle seed/tap bugs.

## Staged build-up

1. **Synthetic LoRa modulator** (`lora/lora_modulator.py`, numpy): generate
   valid up/downchirps and full frames (preamble+sync+SFD+header+payload)
   for known bytes at chosen SF/BW/CR. This is the test oracle — no real
   capture is needed to validate the demod math.
2. **Core symbol demodulator** (`lora/lora_demod.py`): dechirp+FFT+argmax on
   a single symbol window; unit test round-trips modulator-generated symbols
   back to the same integers across all SF 7-12.
3. **Frame sync**: preamble autocorrelation + SFD detection on a *continuous*
   synthetic capture (with leading/trailing noise, arbitrary start offset) to
   find symbol-0 timing without foreknowledge — this is the part most likely
   to need iteration.
4. **Full payload recovery**: Gray-demap → deinterleave → Hamming-decode →
   dewhiten → compare recovered bytes to modulator input, including header
   parsing (length/CR) for explicit mode.
5. **UI wiring**: `LoRaDecoderDialog` + a menu entry in `MainController`,
   reusing the loaded signal's IQ; render decoded bytes as hex + best-effort
   ASCII, and surface FEC-corrected-bit-count / CRC pass-fail as a basic
   confidence indicator.

## Scope boundary — what needs the user

Everything above validates against **synthetic** signals I generate myself,
so stages 1-4 (and initial UI wiring) are fully unattended. What can't be:
validating against a **real** captured LoRa transmission (actual RF
impairments: CFO, SFO, real preamble length quirks from specific vendor
chipsets/gateways). That requires the user to record one with the RSPdx
against a known LoRa device — flagged as a manual verification step for
later, not a blocker for building and self-testing the decoder now.

## Status

- [x] Stage 1 -- synthetic modulator (`src/urh/lora/lora_modulator.py`)
- [x] Stage 2 -- core symbol demod (`lora_chirp.py` + `lora_demod.demod_symbol`);
      round-trips exactly across SF 7-12, OSR 1/2/4 (`tests/lora/test_lora_chirp.py`)
- [x] Stage 3 -- frame sync (`lora_demod.find_frame_start`). Originally a
      per-sample brute-force fine search, which was correct but O(n_sym) FFTs
      per candidate -- impractically slow for large SF/OSR. Replaced with a
      single FFT-based matched filter against the reference upchirp (O(L log L)
      total). Also fixed two real bugs found via testing: (1) the coarse
      autocorrelation stage normalized by one window's energy instead of
      sqrt(E_A*E_B), letting low-energy noise windows produce correlation >1
      and false-positive ahead of the real preamble; (2) "peakiness" as
      peak/sum-of-bins silently gets harder to clear at higher SF for no
      detection-quality reason, since bin count grows with SF -- switched to
      peak/median, which is SF-independent (empirically: noise floor ~3.2-3.5,
      real signal >=14 even at 0dB SNR / SF7).
- [x] Stage 4 -- full payload recovery (header parse, Gray/interleave/Hamming/
      dewhiten). `tests/lora/test_lora_frame_roundtrip.py`: clean round trip
      across all SF(7-12) x CR(1-4) x payload-length combos; noisy (20dB SNR)
      + random-offset round trip across SF x OSR(1/4) x CR; a fuzz test with
      random SF/CR/payload; a 720-case stress grid (SF7-12, BW 125k/500k,
      OSR1/2/4, CR1-4, payload up to 255 bytes) -- all pass, ~4 min runtime.
      17/17 unit tests pass (`python3 -m unittest discover -s tests/lora -v`).
- [x] Stage 5 -- UI wiring. `src/urh/controller/dialogs/LoRaDecoderDialog.py`:
      plain QDialog (no .ui file), takes the currently-loaded Signal's raw IQ
      + user-entered SF/BW/preamble-length/sync-word, runs `decode_frame`,
      shows hex + best-effort ASCII, sync/FEC-error status. Wired into
      `MainController` as "LoRa Decoder..." in the File menu (built and
      inserted in Python code next to `setupUi()`, not added to the .ui
      source, to keep the change additive and not require regenerating
      `ui_main.py`). Smoke-tested end-to-end: constructed a real `Signal` +
      `IQArray` from a synthetic frame, drove the dialog's own decode
      handler, confirmed correct payload/hex/ASCII output through the actual
      UI code path (not just the underlying urh.lora functions). Also
      confirmed `MainController` still constructs cleanly with the new menu
      action present.

Known simplification (documented, not a bug): the header/payload coding here
is a self-consistent from-scratch implementation of the documented LoRa PHY
algorithm (dechirp/FFT, Gray, diagonal interleave, Hamming FEC, whitening),
verified to round-trip against itself, but is **not yet verified bit-exact
against Semtech's reference implementation or a real chipset** (e.g. no
separate header-CRC field, whitening table generated from the stated LFSR
rather than copied from a reference table). Real-hardware validation needs an
actual RSPdx capture of a real LoRa transmission -- flagged in the original
plan as a manual step for the user, not a blocker for this implementation.

Known perf note: per-symbol demod in `decode_frame` is an unvectorized
Python loop (one FFT per symbol). Fine for the test suite (worst case here:
~4 min for 720 cases including large SF/OSR/payload combos) but would be
worth vectorizing before decoding long real captures interactively.
