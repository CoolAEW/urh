import unittest

import numpy as np

from urh.lora.lora_modulator import build_frame
from urh.lora.lora_demod import decode_frame, find_frame_start, LoRaSyncError


def _awgn_capture(rng, frame_iq, snr_db, pre_pad, post_pad):
    noise_power = 10 ** (-snr_db / 10)

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)

    return np.concatenate([noise(pre_pad), frame_iq + noise(len(frame_iq)), noise(post_pad)])


class TestFrameSync(unittest.TestCase):
    def test_finds_frame_at_arbitrary_offset(self):
        rng = np.random.default_rng(1)
        sf, bw = 9, 125000.0
        frame = build_frame(b"sync test", sf, bw, cr=4)
        for offset in (0, 1, 37, 999):
            capture = _awgn_capture(rng, frame, snr_db=20, pre_pad=offset, post_pad=200)
            preamble_start, header_start = find_frame_start(capture, sf, bw, bw)
            self.assertEqual(preamble_start, offset)

    def test_no_signal_raises(self):
        rng = np.random.default_rng(2)
        sf, bw = 8, 125000.0
        noise = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)) * 0.1
        with self.assertRaises(LoRaSyncError):
            find_frame_start(noise, sf, bw, bw)


class TestFrameRoundTrip(unittest.TestCase):
    def test_clean_round_trip_various_params(self):
        payloads = [b"", b"A", b"Hello, LoRa!"]
        for sf in (7, 9, 12):
            for cr in (1, 2, 3, 4):
                for payload in payloads:
                    frame = build_frame(payload, sf, 125000.0, cr=cr)
                    result = decode_frame(frame, sf, 125000.0)
                    self.assertEqual(result["payload"], payload, f"sf={sf} cr={cr} len={len(payload)}")
                    self.assertEqual(result["cr"], cr)
                    self.assertTrue(result["sync_ok"])
                    self.assertEqual(result["uncorrectable_errors"], 0)

    def test_round_trip_with_noise_and_random_offset(self):
        rng = np.random.default_rng(42)
        for sf in (7, 9, 12):
            for osr in (1, 4):
                fs = 125000.0 * osr
                for cr in (1, 2, 3, 4):
                    for payload in (b"", b"A", b"Hello, LoRa!"):
                        frame = build_frame(payload, sf, 125000.0, cr=cr, fs=fs)
                        offset = int(rng.integers(0, 500))
                        capture = _awgn_capture(rng, frame, snr_db=20, pre_pad=offset, post_pad=300)
                        result = decode_frame(capture, sf, 125000.0, fs=fs)
                        self.assertEqual(
                            result["payload"], payload,
                            f"sf={sf} osr={osr} cr={cr} len={len(payload)}",
                        )

    def test_oversampled_round_trip(self):
        sf, bw, osr = 10, 125000.0, 4
        fs = bw * osr
        payload = b"oversampled"
        frame = build_frame(payload, sf, bw, cr=2, fs=fs)
        result = decode_frame(frame, sf, bw, fs=fs)
        self.assertEqual(result["payload"], payload)

    def test_fuzz_random_payloads(self):
        rng = np.random.default_rng(7)
        for _ in range(15):
            sf = int(rng.choice([7, 8, 9, 10]))
            cr = int(rng.choice([1, 2, 3, 4]))
            length = int(rng.integers(0, 40))
            payload = bytes(rng.integers(0, 256, size=length, dtype=np.uint8))
            frame = build_frame(payload, sf, 125000.0, cr=cr)
            result = decode_frame(frame, sf, 125000.0)
            self.assertEqual(result["payload"], payload, f"sf={sf} cr={cr} length={length}")


if __name__ == "__main__":
    unittest.main()
