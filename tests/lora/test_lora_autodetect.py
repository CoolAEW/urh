import unittest

import numpy as np

from urh.lora import lora_autodetect as autodetect
from urh.lora.lora_modulator import build_frame


def _make_snippet(payload, sf, bw, fs, snr_db=20, pre_pad=1000, post_pad=1000, seed=1, cr=4):
    frame = build_frame(payload, sf, bw, cr=cr, fs=fs)
    rng = np.random.default_rng(seed)
    noise_power = 10 ** (-snr_db / 10)

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)

    return np.concatenate([noise(pre_pad), frame + noise(len(frame)), noise(post_pad)])


class TestEstimate(unittest.TestCase):
    def test_ranks_true_combo_at_top(self):
        sf, bw, fs = 9, 125000.0, 500000.0
        payload = b"auto detect this frame"
        snippet = _make_snippet(payload, sf, bw, fs, seed=2)

        candidates = autodetect.estimate(snippet, fs)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual((top.sf, top.bw), (sf, bw))
        self.assertIsNotNone(top.decode_result)
        self.assertEqual(top.decode_result["payload"], payload)

    def test_ranks_true_combo_at_top_different_sf_bw(self):
        # Different point in the grid than the above, to make sure the
        # ranking isn't accidentally tied to one specific combo.
        sf, bw, fs = 10, 250000.0, 1000000.0
        payload = b"a different combo entirely"
        snippet = _make_snippet(payload, sf, bw, fs, seed=3)

        candidates = autodetect.estimate(snippet, fs)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual((top.sf, top.bw), (sf, bw))
        self.assertEqual(top.decode_result["payload"], payload)

    def test_low_bw_high_sf_case_exercises_snippet_pruning(self):
        """Low BW + high SF needs a much longer n_sym at a given native fs
        -- a short snippet may not even contain a full preamble for some
        grid candidates. This should just prune those candidates (via
        preamble_match_score's own LoRaSyncError on a too-short capture),
        not crash or misbehave, and should still find the true combo."""
        sf, bw, fs = 12, 7800.0, 62500.0
        payload = b"low bw"
        snippet = _make_snippet(payload, sf, bw, fs, snr_db=25, pre_pad=500, post_pad=500, seed=4, cr=4)

        candidates = autodetect.estimate(snippet, fs)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual((top.sf, top.bw), (sf, bw))

    def test_no_signal_returns_no_candidates(self):
        rng = np.random.default_rng(5)
        noise = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)) * 0.01
        candidates = autodetect.estimate(noise, 500000.0)
        self.assertEqual(candidates, [])

    def test_never_crashes_across_full_grid_on_garbage(self):
        """Fuzz-robustness: garbage/random bytes-as-IQ shouldn't crash the
        sweep even though most (all) grid candidates will be wrong."""
        rng = np.random.default_rng(6)
        for _ in range(5):
            garbage = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)) * rng.uniform(0.01, 2.0)
            candidates = autodetect.estimate(garbage, 500000.0)
            for c in candidates:
                self.assertGreaterEqual(c.score, 0.0)

    def test_bw_exceeding_sample_rate_is_skipped_not_erroring(self):
        # bw_candidates including a value above fs must not crash --
        # physically unrepresentable, should just be silently skipped.
        sf, bw, fs = 9, 125000.0, 250000.0
        payload = b"bw guard test"
        snippet = _make_snippet(payload, sf, bw, fs, seed=7)

        candidates = autodetect.estimate(
            snippet, fs, bw_candidates=[500000.0, 1000000.0, 125000.0]
        )
        self.assertTrue(candidates)
        self.assertEqual((candidates[0].sf, candidates[0].bw), (sf, bw))


if __name__ == "__main__":
    unittest.main()
