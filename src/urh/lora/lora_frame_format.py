"""
Shared LoRa frame-structure constants used by both the modulator and the
demodulator. Frame layout (in symbol periods):

    [n_preamble upchirps] [2 sync-word upchirps] [2.25 downchirp SFD]
    [header: 8 symbols, rate sf-2, always CR 4/8] [payload: N blocks, rate sf, CR `cr`]

The header carries the payload length (1 byte) and coding rate (1 nibble),
padded with zero nibbles up to `sf - 2` nibbles per block (needed since the
diagonal interleaver operates on fixed-size blocks).
"""

DEFAULT_SYNC_WORD = 0x34
DEFAULT_N_PREAMBLE = 8
SFD_SYMBOLS = 2.25
HEADER_CR = 4
HEADER_SHIFT_BITS = 2  # header uses sf-2 effective bits; shift left by 2 to place in full sf-bit symbol space

# Each sync-word nibble is shifted by this fixed amount regardless of SF when
# placed into a chirp symbol (see lora_modulator.build_frame /
# lora_demod._decode_located_frame). An SF-scaled shift (2**(sf-4)) was tried
# first and seemed more principled, but didn't match any real capture at
# all; the fixed *8 shift produced an *exact* symbol match against a real
# MeshCore frame using its confirmed real sync word (see MESHCORE_SYNC_WORD
# below) -- real hardware's sync-word detection apparently uses a fixed
# shift independent of SF.
SYNC_WORD_SHIFT = 8

# Real sync words, pulled from primary source rather than assumed -- prior
# guesses in this codebase (the generic LoRaWAN public/private defaults,
# 0x34/0x12) happened to include the right MeshCore value but a still-wrong
# comparison formula (see SYNC_WORD_SHIFT above) meant it never actually
# matched anything until both were fixed together.
#   Meshtastic: `const uint8_t syncWord = 0x2b;` in
#     meshtastic/firmware, src/mesh/RadioLibInterface.h
#   MeshCore: uses RadioLib's RADIOLIB_SX126X_SYNC_WORD_PRIVATE (0x12,
#     jgromes/RadioLib, e.g. src/modules/SX126x/SX1262.h) via
#     meshcore-dev/MeshCore's various target.cpp board files.
MESHTASTIC_SYNC_WORD = 0x2B
MESHCORE_SYNC_WORD = 0x12

# Standard LoRa bandwidths (Semtech SX127x/SX126x). Shared between
# LoRaDecoderDialog (manual selection) and lora_autodetect (search space)
# so there's one list, not two drifting copies.
STANDARD_BANDWIDTHS = [
    ("7.8 kHz", 7800),
    ("10.4 kHz", 10400),
    ("15.6 kHz", 15600),
    ("20.8 kHz", 20800),
    ("31.25 kHz", 31250),
    ("41.7 kHz", 41700),
    ("62.5 kHz", 62500),
    ("125 kHz", 125000),
    ("250 kHz", 250000),
    ("500 kHz", 500000),
]

# Spreading factors 7-12 are the full standard LoRa range.
STANDARD_SPREADING_FACTORS = tuple(range(7, 13))

# Named quick-select presets for LoRaDecoderDialog: (label, sf, bw_hz, cr,
# n_preamble, sync_word). `cr` is informational only -- decode_frame reads
# the real coding rate from the frame header -- but included so the field
# shows something sensible. SF/BW pairs mirror
# urh.lora.protocols.identify.MESHTASTIC_PRESETS (kept as a separate table
# rather than importing it: that dict is keyed by (sf, bw) for confidence
# scoring, this is a full parameter tuple for filling in decoder fields, and
# the two data shapes don't share enough to be worth coupling the modules
# over). n_preamble=16 for both: confirmed for Meshtastic from firmware
# research, and empirically confirmed for MeshCore against a real capture
# (2026-08-05 -- the first real capture all session to get sync_ok=True
# used n_preamble=16, not the LoRa-standard default of 8).
LORA_PRESETS = [
    ("Meshtastic - ShortTurbo (SF7/BW500k)", 7, 500000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - ShortFast (SF7/BW250k)", 7, 250000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - ShortSlow (SF8/BW250k)", 8, 250000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - MediumFast (SF9/BW250k)", 9, 250000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - MediumSlow (SF10/BW250k)", 10, 250000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - LongTurbo (SF11/BW500k)", 11, 500000, 4, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - LongFast (SF11/BW250k, default)", 11, 250000, 1, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - LongModerate (SF11/BW125k)", 11, 125000, 4, 16, MESHTASTIC_SYNC_WORD),
    ("Meshtastic - LongSlow (SF12/BW125k, deprecated)", 12, 125000, 4, 16, MESHTASTIC_SYNC_WORD),
    ("MeshCore - EU868 Default (SF8/BW62.5k)", 8, 62500, 4, 16, MESHCORE_SYNC_WORD),
]
