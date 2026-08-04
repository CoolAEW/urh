import unittest

import numpy as np

from urh.lora import lora_chirp as chirp
from urh.lora.lora_demod import demod_symbol


class TestChirpRoundTrip(unittest.TestCase):
    def test_symbol_demod_round_trip(self):
        bw = 125000.0
        for sf in (7, 9, 12):
            n = 1 << sf
            for osr in (1, 2, 4):
                fs = bw * osr
                for s in (0, 1, 5, n // 3, n // 2, n - 1):
                    waveform = chirp.chirp_symbol(s, sf, bw, fs, downchirp=False)
                    peak = demod_symbol(waveform, sf, bw, fs)
                    self.assertEqual(peak, s, f"sf={sf} osr={osr} s={s}")

    def test_waveform_is_unit_magnitude(self):
        waveform = chirp.chirp_symbol(3, 9, 125000.0, 125000.0)
        np.testing.assert_allclose(np.abs(waveform), 1.0, atol=1e-9)

    def test_samples_per_symbol(self):
        self.assertEqual(chirp.samples_per_symbol(7, 125000.0, 125000.0), 128)
        self.assertEqual(chirp.samples_per_symbol(12, 125000.0, 125000.0), 4096)
        self.assertEqual(chirp.samples_per_symbol(7, 125000.0, 500000.0), 512)


if __name__ == "__main__":
    unittest.main()
