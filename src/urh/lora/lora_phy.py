"""
Bit-level LoRa PHY primitives: Gray coding, Hamming (4, 4+cr) FEC, payload
whitening, and the diagonal interleaver.

These are self-consistent building blocks (encode/decode pairs implemented
together and verified to round-trip against each other) rather than a
bit-exact reproduction of Semtech's reference implementation. Real-hardware
bit-for-bit conformance (exact header CRC, exact whitening table, exact
interleaver indexing used by actual chipsets) is not yet verified against a
real capture -- see LORA_PLAN.md.
"""


def gray_encode(value):
    return value ^ (value >> 1)


def gray_decode(value):
    mask = value
    while mask:
        mask >>= 1
        value ^= mask
    return value


def _generate_whitening_table(length=256):
    """8-bit Fibonacci LFSR, polynomial x^8+x^6+x^5+x^4+1, seed 0xFF."""
    table = []
    state = 0xFF
    taps = (7, 5, 4, 3)  # 0-indexed bit positions -> exponents 8,6,5,4
    for _ in range(length):
        table.append(state)
        fb = 0
        for t in taps:
            fb ^= (state >> t) & 1
        state = ((state << 1) | fb) & 0xFF
    return table


WHITENING_TABLE = _generate_whitening_table(256)


def whiten(data_bytes):
    return bytes(b ^ WHITENING_TABLE[i % len(WHITENING_TABLE)] for i, b in enumerate(data_bytes))


def dewhiten(data_bytes):
    # XOR is self-inverse
    return whiten(data_bytes)


def _bits_of(value, width):
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _value_of(bits):
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def hamming_encode_nibble(nibble, cr):
    """nibble: 0-15. cr: 1..4 (coding rate 4/5 .. 4/8). Returns codeword int."""
    d1, d2, d3, d4 = _bits_of(nibble & 0xF, 4)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    if cr == 4:
        bits = [p1, p2, d1, p3, d2, d3, d4]
        overall = 0
        for b in bits:
            overall ^= b
        bits.append(overall)
    elif cr == 3:
        bits = [p1, p2, d1, p3, d2, d3, d4]
    elif cr == 2:
        bits = [p1, p2, d1, d2, d3, d4]
    elif cr == 1:
        bits = [d1, d2, d3, d4, d1 ^ d2 ^ d3 ^ d4]
    else:
        raise ValueError("cr must be 1..4")

    return _value_of(bits)


def hamming_decode_nibble(codeword, cr):
    """Returns (nibble, uncorrectable_error_detected: bool)."""
    rdd = 4 + cr
    bits = _bits_of(codeword & ((1 << rdd) - 1), rdd)

    if cr in (3, 4):
        p1, p2, d1, p3, d2, d3, d4 = bits[:7]
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        error_pos = s1 | (s2 << 1) | (s3 << 2)  # 1..7, 0 = no error

        if cr == 3:
            if error_pos:
                bits[error_pos - 1] ^= 1
            p1, p2, d1, p3, d2, d3, d4 = bits[:7]
            return _value_of([d1, d2, d3, d4]), False
        else:
            p4 = bits[7]
            overall_parity = 0
            for b in bits:
                overall_parity ^= b
            uncorrectable = False
            if error_pos == 0 and overall_parity == 0:
                pass  # no error
            elif error_pos == 0 and overall_parity == 1:
                pass  # error in the lone overall-parity bit; data unaffected
            elif error_pos != 0 and overall_parity == 1:
                bits[error_pos - 1] ^= 1  # single-bit error, correct it
            else:
                uncorrectable = True  # double-bit error: detected, not corrected
            p1, p2, d1, p3, d2, d3, d4 = bits[:7]
            return _value_of([d1, d2, d3, d4]), uncorrectable

    elif cr == 2:
        p1, p2, d1, d2, d3, d4 = bits
        exp_p1 = d1 ^ d2 ^ d4
        exp_p2 = d1 ^ d3 ^ d4
        uncorrectable = (exp_p1 != p1) or (exp_p2 != p2)
        return _value_of([d1, d2, d3, d4]), uncorrectable

    elif cr == 1:
        d1, d2, d3, d4, p = bits
        uncorrectable = (d1 ^ d2 ^ d3 ^ d4 ^ p) != 0
        return _value_of([d1, d2, d3, d4]), uncorrectable

    raise ValueError("cr must be 1..4")


def diagonal_interleave(codewords, ppm, rdd):
    """codewords: list of `ppm` ints, each `rdd` bits wide.
    Returns list of `rdd` ints, each `ppm` bits wide (raw pre-Gray symbol values).
    """
    assert len(codewords) == ppm
    cw_bits = [_bits_of(cw, rdd) for cw in codewords]
    symbols = []
    for k in range(rdd):
        out_bits = [cw_bits[(b - k) % ppm][k] for b in range(ppm)]
        symbols.append(_value_of(out_bits))
    return symbols


def diagonal_deinterleave(symbols, ppm, rdd):
    """Inverse of diagonal_interleave."""
    assert len(symbols) == rdd
    sym_bits = [_bits_of(s, ppm) for s in symbols]
    cw_bits = [[0] * rdd for _ in range(ppm)]
    for k in range(rdd):
        for b in range(ppm):
            row = (b - k) % ppm
            cw_bits[row][k] = sym_bits[k][b]
    return [_value_of(row) for row in cw_bits]
