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
