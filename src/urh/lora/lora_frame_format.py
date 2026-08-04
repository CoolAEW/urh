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
