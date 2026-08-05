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

## Stage 6 -- protocol identification (Meshtastic / MeshCore)

Added after the user asked for the tool to recognize which real-world
protocol produced a decoded LoRa frame (they have both Meshtastic and
MeshCore nodes broadcasting nearby), not just show raw payload bytes.

- [x] `src/urh/lora/protocols/meshtastic.py` -- parses the 16-byte
      unencrypted header, decrypts the payload with the well-known default
      channel PSK via AES-CTR, and manually walks the resulting `Data`
      protobuf's tag stream (no generated protobuf bindings needed) to pull
      out portnum/payload/dest/source. Verified against primary source
      (meshtastic/firmware `Channels.{h,cpp}` for the PSK + key expansion,
      `CryptoEngine.cpp` for the nonce construction, meshtastic/protobufs
      `mesh.proto`/`portnums.proto` for message layout) via `gh api
      search/code` + direct file fetches, not guessed or taken from a single
      third-party source. `tests/lora/test_meshtastic.py`: 7 tests, including
      a full encrypt-then-decrypt round trip and a negative test (wrong key
      must not falsely decrypt).
- [x] `src/urh/lora/protocols/meshcore.py` -- parses the 1-byte
      route/payload-type/version header, optional transport codes, and node
      hash path; fully decodes ADVERT payloads (Ed25519 pubkey/timestamp/
      signature/appdata incl. lat-lon/features/name, with real Ed25519
      signature verification when `cryptography` is available). TXT_MSG/
      GRP_TXT/REQ/RESPONSE/PATH payloads are parsed structurally
      (dest_hash/src_hash/MAC/ciphertext-length) but the ciphertext is
      correctly reported as "encrypted, key unknown" rather than faked --
      decrypting those needs a per-node X25519 shared secret we don't have
      for arbitrary nearby nodes, which is a real limitation, not a bug to
      fix later. Verified against primary source (meshcore-dev/MeshCore
      `docs/packet_format.md` + `docs/payloads.md`); this pass corrected one
      detail from the earlier research summary (hash-size code `0b11` is
      *reserved/invalid*, not a 4-byte hash size). `tests/lora/
      test_meshcore.py`: 12 tests covering header bits, ADVERT (incl.
      tampered-signature rejection), path/transport-code parsing, and the
      TXT_MSG-shaped "don't fabricate plaintext" case.
- [x] `src/urh/lora/protocols/identify.py` -- scores payload bytes (+
      optional SF/BW the frame was demodulated with) against both formats;
      returns a protocol guess with a 0-1 confidence and human-readable
      reasons, not a bare boolean. A pure structural parse alone is
      deliberately not enough to call it identified (`_MIN_CONFIDENCE`) --
      needs SF/BW corroboration and/or a successful decode/decrypt.
      `tests/lora/test_identify.py`: 8 tests, including a 50-iteration
      random-bytes fuzz test asserting the identifier never crashes and
      confidence always stays in [0,1].
- [x] Wired into `LoRaDecoderDialog`: after payload recovery, runs
      `identify.identify()` with the dialog's chosen SF/BW and shows a
      protocol/confidence/fields summary above the existing raw hex/ASCII
      view (which stays, since identification can be wrong).

Full suite: 44/44 tests pass (17 PHY + 27 new protocol tests) --
`python3 -m unittest discover -s tests/lora -v`.

Known limitation carried over from Stage 4/5: none of this has been run
against a real captured frame yet, synthetic-only so far as with the PHY
layer. MeshCore's smaller 1-byte header is a structurally weaker fingerprint
than Meshtastic's 16-byte one -- expect more "unknown"/low-confidence calls
on real MeshCore traffic without SF/BW corroboration than on Meshtastic.

## Real-world validation attempt (live RSPdx capture)

Recorded live IQ with the RSPdx via `tmp_scripts/live_lora_capture.py`
(raw int16 I/Q to file, using the low-level `sdrplay` Cython binding
directly with an `mp.Pipe` in place of URH's full subprocess Device
machinery -- the native callback just needs a `.send_bytes()`-capable
object) and scanned it in overlapping chunks via
`tmp_scripts/decode_capture.py` (a single FFT-based matched filter over an
entire multi-minute capture is impractical -- chunking keeps each
`find_frame_start` call's FFT a manageable size).

- Meshtastic freq (869.525 MHz), SF11/BW250kHz, 240s @ 2 MSPS, gain=45:
  matched filter found a strong energy spike at t=52.5s (peak/noise-floor
  ratio ~50x), but full decode came back `sync_ok=False` with 262
  uncorrectable errors on a 254-byte payload -- essentially noise-quality,
  not a real decode, despite `identify()` guessing "meshtastic" at 0.60
  confidence. Likely not actually a Meshtastic frame (EU868 is a shared
  ISM band), or frame parameters (preamble length, sync word) don't match
  what's really on air.
- MeshCore freq (869.618 MHz), SF8/BW62.5kHz, 90s @ 2 MSPS, gain=45: two
  energy spikes found. The one at t=0 is the interesting result: only
  **4 uncorrectable errors across a 226-byte payload** (cr=3), a
  dramatically better ratio than every other hit in this session (all of
  which were ~50% garbage, i.e. noise) -- but `sync_ok` was still `False`.
  That combination (near-clean payload FEC, failed sync-word check) is a
  specific, actionable lead: it suggests real signal + close-but-not-exact
  timing/offset alignment (sync word symbols landing a fraction off, or a
  wrong assumed sync word constant), not "no signal" or "wrong protocol
  entirely." Worth debugging first on any future pass -- likely a small
  fix in `find_frame_start`'s SFD-to-header offset or the assumed sync
  word rather than a fundamental algorithm problem.
- Neither capture produced a clean `sync_ok=True` decode. Root cause not
  yet isolated -- didn't have the turns budget in this pass to chase the
  near-miss down further.
- Noted but not investigated: a second, concurrent `live_lora_capture.py`
  process (different invocation, not started by this work) was observed
  holding/contending for the SDRplay device partway through, causing a
  `get_devices()` call afterward to return empty. Didn't interfere with
  it, since its origin was unclear -- flagged for the user/parent session
  to check for duplicate/conflicting recording attempts before the next
  live-capture pass.

Raw captures (`tmp_scripts/captures/*.iq`, ~1.9GB + ~730MB) are
intentionally untracked/not committed -- large binary, easily
regenerated. The two scripts are committed since they're reusable for the
next attempt.

## Second real-world pass (same session, continued)

Fixed a real bug found while scanning: `_demod_symbols` could silently
truncate a symbol window near a chunk boundary and then fail to broadcast
against the full-length reference chirp instead of raising
`LoRaSyncError` cleanly (commit `48cdfeff`, all 44 tests still pass
after).

Re-ran a shorter (60s) Meshtastic-frequency capture at lower gain (35,
down from 45) to test the clipping hypothesis, timed against the user
actually sending a live Meshtastic message partway through the window.
Two findings that change the read on this problem:

1. **Peak amplitude barely moved (~1.07, still at/above full digital
   scale) despite a 10dB manual gain reduction.** A real analog-frontend
   clipping fix should have cut that peak by roughly 3x. It didn't --
   which points at the SDRplay API's internal AGC still being active
   underneath the manual `gRdB` we set (see `calculate_gain_reduction` in
   `sdrplay.pyx` -- worth checking whether `sdrplay_api` needs an
   explicit AGC-disable call, not just a gain value, since the current
   `init_stream` never touches an AGC control field). **Simply turning
   the gain knob further is unlikely to fix this on its own.**
2. **The real signal spike (~t=25s, matching the user's actual send
   window) produced no frame-sync candidate at all** -- the detector
   didn't recognize it as a LoRa preamble.
3. **The only "decode" candidate in this run was at t=0.0s again** --
   same as the earlier "4-error near-miss" MeshCore result from the first
   pass, which was also at t=0.0s. Two different captures, two different
   frequencies/SF/BW, both producing a spurious sync-detector trigger
   right at recording start. **This reframes that earlier near-miss: it
   was very likely a capture-startup artifact (buffer priming / filter
   settling transient), not a near-successful real decode as first
   thought.** Any future pass should discard the first chunk of each
   capture, or start `find_frame_start` scanning some margin after
   stream-open, before trusting a t≈0 hit.

## Status: paused for user to continue hands-on

No clean real-world decode yet. Open leads for whoever picks this up
next, in priority order: (a) check/disable SDRplay AGC explicitly rather
than assuming manual `gRdB` alone controls level, (b) trim/ignore t≈0 of
each capture as a startup-artifact zone, (c) the original CFO/timing-
offset compensation gap -- synthetic tests never exercised real
oscillator drift or Doppler/frequency error, which real hardware always
has and the current `find_frame_start`/`demod_symbol` chain has no
correction for.

Handy commands for further manual testing:
```
# record N seconds at a frequency (defaults: 2 MSPS, 300kHz IF bw)
cd ~/src/urh-sdrplay-v3
PYTHONPATH=src python3 tmp_scripts/live_lora_capture.py \
  --freq 869525000 --duration 60 --out tmp_scripts/captures/out.iq --gain 35

# scan a capture for a decodable frame
PYTHONPATH=src python3 tmp_scripts/decode_capture.py \
  --path tmp_scripts/captures/out.iq --sf 11 --bw 250000 --fs 2000000

# or use the full GUI (File -> LoRa Decoder...) on a signal recorded/loaded normally
PYTHONPATH=src python3 src/urh/main.py
```

## Robustness/speed/auto-detect/UI follow-up (2026-08-05)

Real-world testing (see above) surfaced enough real gaps that a proper follow-up
plan was written and approved (`~/.claude/plans/soft-skipping-rain.md`, 4 phases:
robustness/confidence scoring, performance dedup, SF/BW auto-detect, table UI +
context-menu entry point). Implementing in that order.

### Phase 1: peakiness-gated demod + composite confidence score -- DONE

Root cause finally confirmed for the recurring "clean-but-implausible" artifacts
seen across multiple real captures (e.g. payload header byte `0xFF`): once real
signal ends mid-frame, the decoder kept blindly demodulating the declared payload
length into trailing noise. That noise deterministically dechirps to a low-
diversity, low-peakiness symbol run rather than random values, so it can pass
Hamming FEC with a deceptively *low* error count -- a false-positive "clean"
decode. Verified directly against a real capture: peakiness dropped from 3-6.35
to a hard 0.0 cliff mid-frame, exactly where real content stopped.

Fix, in `lora_demod.py`:
- Unified the FFT path: `_demod_symbol_with_peakiness()` does one `_dechirp_fft`
  call per symbol and derives both the bin value and peakiness from the same
  magnitude array (previously `demod_symbol`/`_symbol_peakiness` were separate,
  and `_symbol_peakiness` was actually dead code -- defined, never called).
  `_demod_symbols()` now computes the reference downchirp once before its loop
  instead of once per symbol (a real, separate perf bug caught in the process)
  and returns `(symbols, peakiness_list, new_offset)`.
- Mid-frame collapse detection: per-frame reference peakiness = median of the
  header symbols' peakiness (not a fixed constant -- SNR varies across
  captures); a payload symbol is "collapsed" if its peakiness drops below
  `PAYLOAD_COLLAPSE_FRACTION` (0.3, starting guess) of that reference, and a
  *sustained* run of >= `PAYLOAD_COLLAPSE_RUN_LENGTH` (4) consecutive collapsed
  symbols truncates the payload to what decoded before the collapse rather than
  failing outright -- a truncated Meshtastic/MeshCore header can still
  corroborate a protocol guess even without the full payload. Result dict gains
  `payload_truncated`/`truncated_at_symbol`.
- Composite `confidence` score (0-1): weighted combination of error rate (not
  raw count), payload-peakiness consistency (the strongest signal -- this is
  the one that would have caught the `0xFF` artifact), and demodulated-symbol
  entropy (weighted lower -- legitimate payloads can have genuinely low-entropy
  stretches, e.g. padding). `sync_ok=False` is a multiplicative penalty
  (`x0.3`), not a hard gate, since a sync-word mismatch with otherwise-clean FEC
  has already proven to be a real, worth-investigating signal in real captures
  (see the "second real-world pass" section above). Weights are named
  module-level constants for easy retuning -- **not yet validated against real
  captures, only synthetic**, same caveat as CFO correction's initial rollout.
- `identify()` gained optional `decode_confidence`/`payload_truncated` params
  that scale (never raise) its score, so a truncated/uncertain payload can't
  outscore a clean one purely on structural pattern match. Backward compatible
  (defaults preserve old behavior).
- `LoRaDecoderDialog` now sorts hits by `confidence` and surfaces truncation in
  the result text.

**A real bug was found and fixed during this phase's own testing**: the initial
peakiness metric (peak-bin-magnitude / median-bin-magnitude) swung wildly
(~300x) between symbols of the *same clean, noiseless synthetic frame*, because
with near-zero real noise the non-peak FFT bins are dominated by float64
rounding residue rather than anything physical -- broke 2 existing tests
(`test_clean_round_trip_various_params`, `test_fuzz_random_payloads`) via
false-positive collapse detection on legitimate short payloads. Fixed by
flooring the median to a small fraction of the peak
(`max(median, peak_mag * 1e-6)`) before dividing, which stabilizes the metric in
the noiseless/very-high-SNR regime without affecting genuinely noisy/collapsed
windows (where the floor isn't the binding constraint). Verified the fix against
both the previously-broken clean cases and a realistic spliced-trailing-silence
scenario (`tests/lora/test_lora_confidence.py`) -- collapse detection correctly
fires only on the latter.

64/64 tests pass (`PYTHONPATH=src python3 -m unittest discover -s tests/lora`).

### Phase 2: performance dedup -- DONE

Split `decode_frame` into `_locate_frame` (find_frame_start + CFO-iteration --
depends on `n_preamble` but not `sync_word`) and `_decode_located_frame`
(sync-word check + header + payload decode -- the part that depends on
`sync_word`). `decode_frame` itself is now a thin wrapper preserving its exact
prior signature/return dict, so all pre-existing tests pass unmodified
(behavior-preserving refactor, verified by the full suite staying green).

`scan_chunk_for_frames` now loops `n_preamble` outer (calling `_locate_frame`
once, catching `LoRaSyncError` once) and `sync_word` inner (calling
`_decode_located_frame` against the already-located/CFO-corrected IQ) --
eliminates the redundant re-location work every `(n_preamble, sync_word)` pair
previously did independently. Also cached `reference_upchirp`/
`reference_downchirp` in `lora_chirp.py` via `functools.lru_cache` (verified no
caller mutates the returned array in place -- all usages multiply against it,
which allocates a new array).

Measured with a micro-benchmark (`tmp_scripts/bench_scan_chunk.py`, not
committed as a test) on a representative chunk with the default 2 n_preamble x
2 sync_word sweep: 284ms/call before -> 143ms/call after, ~2x, matching the
expected reduction from eliminating the sync_word fan-out redundancy.

New regression test (`test_lora_scan.py`): asserts `_locate_frame` is called
exactly `len(n_preamble_options)` times (not `x len(sync_word_options)`) per
chunk via `unittest.mock.patch` call-counting.

65/65 tests pass.

Cython porting of the hot `_dechirp_fft` inner loop remains explicitly out of
scope (per the approved plan) -- only worth pursuing if a future real-capture
pass shows scan latency is still a problem after this dedup.

### Phase 3: SF/BW auto-detect -- DONE

New module `src/urh/lora/lora_autodetect.py` (pure Python/numpy, no PyQt
dependency, mirroring `AutoInterpretation.estimate()`'s independence from
`Signal`/Qt). Two-stage cost control:
1. Cheap: score every (SF, BW) combination in the standard grid (6 SF x 10
   BW = 60 combos) using a newly-factored-out `preamble_match_score()`
   (pulled out of `find_frame_start`, which now calls it internally too --
   single source of truth) -- the peak-to-noise-floor ratio it already
   computed internally and, until now, just thresholded and discarded.
   Low-BW candidates are decimated (plain numpy boxcar-average, no scipy
   available in this environment) toward a ~4x oversampling ratio before
   matched-filtering, so a low-BW candidate doesn't run a huge symbol
   length against a high native sample rate. No FEC/payload decode
   attempted at this stage.
2. Expensive: full `decode_frame` attempts (via `scan_chunk_for_frames`,
   trying the same small n_preamble/sync_word candidate sets manual
   scanning already does) only on the top-K (3) cheap-stage winners, to
   confirm/select the final answer and opportunistically recover
   n_preamble/sync_word too.

`LoRaDecoderDialog` gained an "Auto-detect" button (background `QThread` via
new `_AutoDetectWorker`, reusing the existing progress bar/status label,
indeterminate progress since the grid sweep has no natural per-step hook)
that scans the first 3s of the loaded signal and writes the winning
candidate's SF/BW/preamble-length/sync-word directly into the same
manually-editable fields a user would set by hand -- mirrors
`Signal.auto_detect()`'s "auto-detect writes into the same fields" pattern,
scoped to this dialog's own widgets (no `Signal.py` changes needed).

`LoRaDecoderDialog.BANDWIDTHS` and a new `STANDARD_SPREADING_FACTORS` moved to
`lora_frame_format.py` as shared constants (`STANDARD_BANDWIDTHS`) so the
dialog and the autodetect search space reference one list.

Smoke-tested end-to-end through the real QThread/Qt event loop (same pattern
as prior UI smoke tests): a synthetic SF9/125kHz signal was correctly
identified and its parameters written into the actual spinbox/combobox
widgets, not just the underlying `estimate()` function.

New test file `test_lora_autodetect.py` (6 tests): ranks the true SF/BW combo
at the top across two different grid points, a low-BW/high-SF case to
exercise snippet-length pruning, no-signal returns no candidates, fuzz-
robustness across the full grid on garbage input, and a bw-exceeds-sample-
rate guard. 71/71 tests pass overall.

**Flagged as exploratory** (same caveat as CFO correction and Phase 1's
collapse thresholds): the 3-second snippet length and top-K=3 defaults are a
defensible starting point, not validated against real captures yet -- real
low-SNR or unusual-preset traffic may need these retuned.

### Phase 4: table UI + context-menu entry point -- DONE

**Context-menu entry point**: `SignalFrame` gained a `lora_decode_requested =
pyqtSignal(Signal)` and a new "LoRa Decode..." action in `contextMenuEvent`
(right-click a loaded signal), following the exact existing pattern the
"Auto-Detect signal parameters" action already used. Bubbled up through
`SignalTabController` by connecting it inside the *shared* private
`__create_connects_for_signal_frame` helper (already called by both
`add_signal_frame` and `add_empty_frame`, confirmed by reading both call
sites) so both signal-construction paths get the wiring from one connection,
not two. `MainController.on_show_lora_decoder_dialog_action_triggered` (the
existing File-menu handler) was refactored into a shared
`open_lora_decoder_dialog(signal)` plus two thin callers: the File-menu slot
(still defaults to the first loaded signal, preserving old behavior) and a
new `on_lora_decode_requested(signal)` slot wired to the relayed context-menu
signal.

**Table UI**: `LoRaDecoderDialog.result_view` (a single `QPlainTextEdit`
dumping every hit as concatenated text) replaced with `hits_table`
(`QTableWidget`, one row per scan hit: Time, Confidence, Sync OK, Protocol,
CR, FEC errors, Truncated, Payload preview -- sorted by confidence descending,
same as before) plus a `detail_view` (`QPlainTextEdit`, unchanged formatting
via `_format_result`/`_format_identification`) below it showing the full
hex/ASCII/identification breakdown for whichever row is selected (auto-
selects the top row after a scan). Styling (`setAlternatingRowColors`,
header resize behavior) mirrors `tableWidgetPreview` in `ui_csv_wizard.py`,
the closest existing lightweight `QTableWidget` reference in the codebase --
deliberately not the heavier `QAbstractTableModel`/`TableView.py`/
`LabelValueTableModel` machinery built around `ProtocolAnalyzer`, per the
resolved lighter-touch-integration decision. Auto-detect's summary now also
goes to `detail_view` (it isn't a scan-hit list, so it doesn't populate the
table).

Smoke-tested end-to-end through real Qt: constructed `MainController`,
confirmed the `SignalTabController -> MainController` signal relay is wired
(`receivers() > 0`), and directly drove `on_lora_decode_requested` with a
synthetic `Signal` to confirm the dialog opens without error via that path.
Separately drove the dialog's own `on_decode_clicked` through the real event
loop and confirmed the table populates with correct per-column values, a row
auto-selects, and the detail pane updates to match.

No new automated tests for this phase (pure UI wiring; the project has no Qt
test harness beyond the manual/smoke-test pattern already used for the
dialog's earlier stages -- `pytest-qt` isn't installed, `unittest`-only).
71/71 existing tests still pass unmodified.

## All four planned phases (robustness, performance, auto-detect, UI) are
now complete. See `~/.claude/plans/soft-skipping-rain.md` for the original
approved plan this section implements.

## Sync-word breakthrough (2026-08-05) -- root cause of "never identifies anything"

User reported the decoder couldn't identify signal type on real captures.
Root-caused to two compounding bugs, neither of which synthetic tests could
ever catch (both sides of the modulator/demodulator shared the same wrong
assumptions, so round-trip tests stayed self-consistently green throughout):

1. **Wrong sync-word symbol formula.** Earlier this session, the sync-word
   nibble shift was "fixed" from a hardcoded `*8` to `2**(sf-4)`, reasoning
   from symbol-space proportions. That reasoning was wrong. Reverted to the
   fixed `*8` shift -- confirmed against a real MeshCore capture using its
   *actual* sync word (see #3 below): the fixed shift produced an **exact**
   symbol match (`[8, 16]` observed, `[8, 16]` expected, bit-for-bit) against
   real hardware. The SF-scaled version never matched anything except by
   coincidence at SF7.

2. **Exact-equality sync check.** `sync_ok` compared demodulated symbols to
   the expected value with `==`. A real captured symbol carries residual
   CFO/timing noise and essentially never lands on the exact expected bin,
   so this check failed 100% of the time regardless of whether the sync word
   guess was even correct. Changed to nearest-nibble comparison (same
   principle `demod_symbol`'s own `argmax` already uses elsewhere).

3. **Sync words were never verified against primary source.** All session,
   0x34/0x12 (generic LoRaWAN public/private defaults) were used as guesses.
   Checked GitHub source directly (`gh search code`) instead:
   - Meshtastic: `const uint8_t syncWord = 0x2b;` --
     `meshtastic/firmware`, `src/mesh/RadioLibInterface.h`.
   - MeshCore: uses RadioLib's `RADIOLIB_SX126X_SYNC_WORD_PRIVATE` (= `0x12`,
     confirmed in `jgromes/RadioLib`) via various `target.cpp` board files in
     `meshcore-dev/MeshCore`.
   Both are now named constants (`MESHTASTIC_SYNC_WORD`, `MESHCORE_SYNC_WORD`)
   in `lora_frame_format.py`. Notably, 0x12 (MeshCore's real value) was
   already one of the two guesses being tried all along -- it just could
   never match because of bugs #1 and #2 above.

**Result**, re-scanning all 9 accumulated real captures with the fix:
`sync_ok=True` now fires on 10/144 hits (0/144 before, all session). Best
hit (t=167.5s in a 5-minute capture, 1 uncorrectable error on a 49-byte
payload): confidence 0.28 -> 0.93, and `identify()` flips from `unknown`
to `meshcore` at 69% confidence. This is the direct fix for the reported
"can't identify signal type" complaint. Full suite: 71/71 still pass.

**New lead, not yet resolved:** two of the highest-confidence `sync_ok=True`
hits (t=167.5s and t=42.5s in the same capture) both show implausibly large
`hash_count` values (16 and 24 hops) in the MeshCore path-length byte
specifically -- both otherwise decode cleanly (1-3 errors) and now pass
sync validation, but `meshcore.parse()` fails on the path length being
inconsistent with the remaining payload size. Worth investigating whether
this is a residual decode issue concentrated in that specific byte position,
or a real protocol detail not yet understood (e.g. a different path
encoding for TRANSPORT_DIRECT routes specifically -- both anomalous hits
have `route_type=3`).

## Presets, and an ADVERT hunt that didn't land (2026-08-05, session wrap-up)

Added a "Preset" quick-select dropdown to `LoRaDecoderDialog` (commit
`4c82b887`): 9 Meshtastic modem presets + MeshCore's EU868 default, using
`LORA_PRESETS` in `lora_frame_format.py` (sf/bw/cr/n_preamble/sync_word per
entry). Picking one fills in all the manual fields at once instead of
requiring the user to know/enter them by hand.

Then tried, live, to specifically capture and decode a MeshCore ADVERT
packet (the self-authenticating type with node name/pubkey/signature) --
none of the 10 confirmed real `sync_ok=True` hits so far are ADVERT type
(payload types seen: TXT_MSG, ACK, GRP_TXT, MULTIPART, TRACE). Three
targeted attempts, none landed cleanly:
1. User sent multiple zero-hop + flood adverts in one window -- real energy
   present for ~30s but scattered/broadband, no clean matched-filter lock.
   Leading hypothesis: flood routing causes multiple nearby nodes to
   rebroadcast near-simultaneously, colliding at the RF level (not
   something the decoder can fix).
2. Single zero-hop advert only, gain=125 (0.87dB less than the previous
   attempt) -- energy scattered across many separate ~1s windows rather
   than one clean burst, zero hits.
3. Single zero-hop advert, gain=0 (user's own proven-successful setting
   from their independent GUI recordings, notably lower than the gain=125
   this session's own capture script had been defaulting to all along) --
   only a brief, weak energy blip (0.072 vs 0.017 background, ~4x -- likely
   too weak/short to clear the matched filter's noise-floor gate), zero
   hits.

**Not yet resolved**: whether this is send/record timing lag (the
back-and-forth between telling the user to send and the recording already
being live), genuine signal weakness at lower gain, or something
ADVERT-specific. `tmp_scripts/show_latest_run.py` (committed) is a working,
OOM-safe example of loading a real capture + opening the dialog +
exercising the preset dropdown through its real Qt code path -- useful
starting point for the next attempt, and documents a real gotcha (calling
`MainController.add_signal()` on a multi-hundred-MB raw capture OOM-killed
this 7.5GB-RAM machine by running URH's native wavelet-based auto-detect
over the whole array; load the signal directly and skip `add_signal()` for
large LoRa captures).

**Next steps when resuming**: (a) try a longer passive capture (10+ min,
no manual trigger) to catch an ADVERT on its own periodic schedule instead
of racing live send timing; (b) resolve the path-length/hash_count anomaly
noted above, which affects two of our best non-ADVERT hits too; (c) revisit
gain choice systematically (a short sweep across a few gain values against
the same known-good send, rather than one-shot guesses) now that we know
the user's own successful captures used much lower gain than this
session's scripts had been defaulting to.
