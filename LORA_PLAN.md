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
Research/scoping complete. Implementation starts at stage 1.
