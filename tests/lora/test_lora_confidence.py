import unittest

import numpy as np

from urh.lora.lora_modulator import build_frame
from urh.lora.lora_demod import decode_frame, chirp


def _awgn(rng, frame_iq, snr_db, pre_pad, post_pad):
    noise_power = 10 ** (-snr_db / 10)

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)

    return np.concatenate([noise(pre_pad), frame_iq + noise(len(frame_iq)), noise(post_pad)])


def _splice_trailing_silence(rng, frame_iq, sf, bw, fs, cutoff_payload_symbol, snr_db, pre_pad, post_pad):
    """Build a capture that's a real frame up through `cutoff_payload_symbol`
    payload symbols, then pure noise (silence) for the rest -- the
    empirically-observed real-world "short transmission followed by
    trailing noise" pattern, rather than generic AWGN over the whole frame."""
    n_sym = chirp.samples_per_symbol(sf, bw, fs)
    from urh.lora.lora_demod import find_frame_start

    _, header_start = find_frame_start(frame_iq, sf, bw, fs)
    # header is always 8 symbols (rdd_hdr = 4 + HEADER_CR = 4+4)
    payload_start = header_start + 8 * n_sym
    cutoff = payload_start + cutoff_payload_symbol * n_sym

    noise_power = 10 ** (-snr_db / 10)

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)

    noisy = frame_iq + noise(len(frame_iq))
    noisy = np.concatenate([noisy[:cutoff], noise(len(noisy) - cutoff)])
    return np.concatenate([noise(pre_pad), noisy, noise(post_pad)])


class TestPayloadCollapseDetection(unittest.TestCase):
    def test_trailing_collapse_truncates_and_flags(self):
        sf, bw = 9, 125000.0
        payload = b"this is a longer test payload for truncation testing"
        frame = build_frame(payload, sf, bw, cr=4)
        rng = np.random.default_rng(100)
        capture = _splice_trailing_silence(
            rng, frame, sf, bw, bw, cutoff_payload_symbol=12, snr_db=20, pre_pad=300, post_pad=300
        )

        result = decode_frame(capture, sf, bw, fs=bw)
        self.assertTrue(result["payload_truncated"])
        self.assertIsNotNone(result["truncated_at_symbol"])
        self.assertEqual(result["truncated_at_symbol"], 12)
        # the correctly-decoded prefix must still be correct, byte for byte
        self.assertEqual(result["payload"], payload[: len(result["payload"])])
        self.assertGreater(len(result["payload"]), 0)
        self.assertLess(len(result["payload"]), len(payload))

    def test_truncated_decode_has_lower_confidence_than_clean(self):
        sf, bw = 9, 125000.0
        payload = b"this is a longer test payload for truncation testing"
        frame = build_frame(payload, sf, bw, cr=4)

        rng_clean = np.random.default_rng(101)
        clean_capture = _awgn(rng_clean, frame, snr_db=20, pre_pad=300, post_pad=300)
        clean_result = decode_frame(clean_capture, sf, bw, fs=bw)

        rng_trunc = np.random.default_rng(102)
        trunc_capture = _splice_trailing_silence(
            rng_trunc, frame, sf, bw, bw, cutoff_payload_symbol=12, snr_db=20, pre_pad=300, post_pad=300
        )
        trunc_result = decode_frame(trunc_capture, sf, bw, fs=bw)

        self.assertFalse(clean_result["payload_truncated"])
        self.assertTrue(trunc_result["payload_truncated"])
        self.assertLess(trunc_result["confidence"], clean_result["confidence"])

    def test_no_collapse_on_clean_signal_across_sf_range(self):
        """Regression guard for the false-positive bug found during
        implementation: a perfectly clean (near-zero-noise) synthetic
        signal has near-zero real noise floor, which made an early,
        unfloored peakiness metric swing wildly (~300x between symbols of
        the same frame from float64 rounding alone) and falsely triggered
        collapse detection. Covers every SF since the failure was SF/CR
        dependent (shorter payloads at low SF have fewer symbols relative
        to the run-length, making false positives easier to trigger)."""
        payload = b"regression payload"
        for sf in range(7, 13):
            frame = build_frame(payload, sf, 125000.0, cr=4)
            result = decode_frame(frame, sf, 125000.0)
            self.assertFalse(result["payload_truncated"], f"sf={sf}")
            self.assertEqual(result["payload"], payload, f"sf={sf}")

    def test_short_run_of_bad_symbols_does_not_truncate(self):
        """A handful of below-threshold symbols shorter than
        PAYLOAD_COLLAPSE_RUN_LENGTH (momentary fade, not real collapse)
        should not truncate an otherwise-good frame."""
        sf, bw = 9, 125000.0
        payload = b"momentary fade should not truncate this payload"
        frame = build_frame(payload, sf, bw, cr=4)
        rng = np.random.default_rng(103)
        # Moderate, uniform noise throughout (no real collapse) -- should
        # decode cleanly and not falsely truncate.
        capture = _awgn(rng, frame, snr_db=15, pre_pad=300, post_pad=300)
        result = decode_frame(capture, sf, bw, fs=bw)
        self.assertFalse(result["payload_truncated"])
        self.assertEqual(result["payload"], payload)


class TestConfidenceScore(unittest.TestCase):
    def test_confidence_in_unit_range_fuzz(self):
        rng = np.random.default_rng(200)
        for _ in range(50):
            sf = int(rng.choice([7, 8, 9, 10, 11, 12]))
            cr = int(rng.choice([1, 2, 3, 4]))
            length = int(rng.integers(0, 60))
            payload = bytes(rng.integers(0, 256, size=length, dtype=np.uint8))
            frame = build_frame(payload, sf, 125000.0, cr=cr)
            snr_db = float(rng.uniform(0, 25))
            capture = _awgn(rng, frame, snr_db=snr_db, pre_pad=int(rng.integers(0, 400)), post_pad=300)
            try:
                result = decode_frame(capture, sf, 125000.0)
            except Exception:
                continue  # low-SNR sync failures are expected/fine here
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_sync_fail_penalizes_confidence(self):
        """A decode with sync_ok=False should score lower than an
        otherwise-identical clean decode, even though it isn't hard-gated
        to zero (a promising-but-unconfirmed sync-word mismatch has proven
        to be a real, worth-investigating signal in real captures)."""
        sf, bw = 9, 125000.0
        payload = b"sync word mismatch test"
        frame = build_frame(payload, sf, bw, cr=4)
        result_ok = decode_frame(frame, sf, bw, sync_word=0x34)
        result_bad_sync = decode_frame(frame, sf, bw, sync_word=0x12)

        self.assertTrue(result_ok["sync_ok"])
        self.assertFalse(result_bad_sync["sync_ok"])
        self.assertLess(result_bad_sync["confidence"], result_ok["confidence"])
        # still not zeroed out -- a real, meaningful decode with mismatched
        # sync word retains a nonzero confidence rather than being hidden.
        self.assertGreater(result_bad_sync["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
