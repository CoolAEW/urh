import random
import unittest

from urh.lora import lora_phy as phy


class TestGrayCode(unittest.TestCase):
    def test_round_trip(self):
        for v in range(0, 1 << 12):
            self.assertEqual(phy.gray_decode(phy.gray_encode(v)), v)


class TestWhitening(unittest.TestCase):
    def test_self_inverse(self):
        data = bytes(range(256)) * 2
        self.assertEqual(phy.dewhiten(phy.whiten(data)), data)

    def test_actually_changes_data(self):
        data = bytes([0x00] * 32)
        self.assertNotEqual(phy.whiten(data), data)


class TestHamming(unittest.TestCase):
    def test_round_trip_no_errors(self):
        for cr in (1, 2, 3, 4):
            for nibble in range(16):
                codeword = phy.hamming_encode_nibble(nibble, cr)
                decoded, err = phy.hamming_decode_nibble(codeword, cr)
                self.assertEqual(decoded, nibble)
                self.assertFalse(err)

    def test_single_bit_correction_cr3_cr4(self):
        for cr in (3, 4):
            rdd = 4 + cr
            for nibble in range(16):
                codeword = phy.hamming_encode_nibble(nibble, cr)
                for bitpos in range(rdd):
                    flipped = codeword ^ (1 << bitpos)
                    decoded, _ = phy.hamming_decode_nibble(flipped, cr)
                    self.assertEqual(decoded, nibble, f"cr={cr} nibble={nibble} bit={bitpos}")

    def test_double_bit_error_detected_cr4(self):
        # cr=4 is SECDED: a double-bit error must be flagged uncorrectable,
        # not silently mis-corrected.
        detected_any = False
        for nibble in range(16):
            codeword = phy.hamming_encode_nibble(nibble, 4)
            for b1 in range(8):
                for b2 in range(b1 + 1, 8):
                    flipped = codeword ^ (1 << b1) ^ (1 << b2)
                    _, err = phy.hamming_decode_nibble(flipped, 4)
                    if err:
                        detected_any = True
        self.assertTrue(detected_any)

    def test_detect_only_cr1_cr2(self):
        for cr in (1, 2):
            rdd = 4 + cr
            for nibble in range(16):
                codeword = phy.hamming_encode_nibble(nibble, cr)
                for bitpos in range(rdd):
                    flipped = codeword ^ (1 << bitpos)
                    _, err = phy.hamming_decode_nibble(flipped, cr)
                    self.assertTrue(err, f"cr={cr} nibble={nibble} bit={bitpos} should be flagged")


class TestInterleaver(unittest.TestCase):
    def test_round_trip(self):
        rng = random.Random(0)
        for ppm in (5, 7, 10, 12):
            for cr in (1, 2, 3, 4):
                rdd = 4 + cr
                codewords = [rng.getrandbits(rdd) for _ in range(ppm)]
                symbols = phy.diagonal_interleave(codewords, ppm, rdd)
                self.assertEqual(len(symbols), rdd)
                back = phy.diagonal_deinterleave(symbols, ppm, rdd)
                self.assertEqual(back, codewords)


if __name__ == "__main__":
    unittest.main()
