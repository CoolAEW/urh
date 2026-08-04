import unittest
from unittest import mock

import numpy as np

from urh.lora import lora_demod
from urh.lora.lora_demod import scan_for_frames, scan_chunk_for_frames
from urh.lora.lora_modulator import build_frame


def _awgn_capture(rng, frame_iq, snr_db, pre_pad, post_pad):
    noise_power = 10 ** (-snr_db / 10)

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)

    return np.concatenate([noise(pre_pad), frame_iq + noise(len(frame_iq)), noise(post_pad)])


class TestScanForFrames(unittest.TestCase):
    def test_finds_frame_spanning_multiple_chunks(self):
        rng = np.random.default_rng(7)
        sf, bw = 8, 125000.0
        payload = b"chunked scan test"
        frame = build_frame(payload, sf, bw, cr=4)
        self.assertLess(len(frame), 18750, "test assumes frame fits within one overlap window")
        # Offset the frame several chunks in, so the scan has to get past a
        # few empty chunks before reaching the one containing it. Overlap
        # must exceed the frame length so it can't be split across chunks.
        offset = 40000
        capture = _awgn_capture(rng, frame, snr_db=20, pre_pad=offset, post_pad=20000)

        hits, max_mags = scan_for_frames(
            capture, sf, bw, bw, chunk_sec=0.3, overlap_sec=0.15
        )

        self.assertTrue(any(h["sync_ok"] and h["payload"] == payload for h in hits))
        self.assertTrue(len(max_mags) > 1)

    def test_progress_callback_reports_monotonic_chunks(self):
        rng = np.random.default_rng(8)
        sf, bw = 7, 125000.0
        frame = build_frame(b"progress", sf, bw, cr=4)
        capture = _awgn_capture(rng, frame, snr_db=20, pre_pad=1000, post_pad=1000)

        seen = []
        scan_for_frames(
            capture, sf, bw, bw, chunk_sec=0.05, overlap_sec=0.01,
            progress_cb=lambda i, total, t: seen.append((i, total)),
        )

        self.assertTrue(len(seen) > 1)
        indices = [i for i, _ in seen]
        self.assertEqual(indices, sorted(indices))
        totals = {total for _, total in seen}
        self.assertEqual(len(totals), 1)  # total_chunks is stable across calls
        self.assertTrue(max(indices) < next(iter(totals)))

    def test_should_stop_aborts_early(self):
        rng = np.random.default_rng(9)
        sf, bw = 7, 125000.0
        frame = build_frame(b"stop me", sf, bw, cr=4)
        capture = _awgn_capture(rng, frame, snr_db=20, pre_pad=1000, post_pad=20000)

        calls = {"n": 0}

        def should_stop():
            calls["n"] += 1
            return calls["n"] > 2

        hits, max_mags = scan_for_frames(
            capture, sf, bw, bw, chunk_sec=0.05, overlap_sec=0.01,
            should_stop=should_stop,
        )
        self.assertLessEqual(len(max_mags), 3)

    def test_no_signal_returns_no_hits(self):
        rng = np.random.default_rng(10)
        noise = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)) * 0.01
        hits, max_mags = scan_for_frames(noise, 8, 125000.0, 125000.0, chunk_sec=0.05)
        self.assertEqual(hits, [])
        self.assertTrue(len(max_mags) > 1)


class TestScanChunkDedupesLocation(unittest.TestCase):
    def test_locate_frame_called_once_per_n_preamble_not_per_sync_word(self):
        """Performance regression guard: sync_word doesn't affect frame
        location, only the post-location sync-word-symbol comparison, so
        _locate_frame (the expensive find_frame_start + CFO-iteration work)
        should run once per n_preamble value, not once per
        (n_preamble, sync_word) combination."""
        rng = np.random.default_rng(11)
        sf, bw = 8, 125000.0
        frame = build_frame(b"dedup test", sf, bw, cr=4)
        capture = frame + (rng.standard_normal(len(frame)) + 1j * rng.standard_normal(len(frame))) * 0.05

        n_preamble_options = (8, 16)
        sync_word_options = (0x34, 0x12, 0x99)

        with mock.patch(
            "urh.lora.lora_demod._locate_frame", wraps=lora_demod._locate_frame
        ) as spy:
            scan_chunk_for_frames(capture, sf, bw, bw, n_preamble_options, sync_word_options)
            self.assertEqual(spy.call_count, len(n_preamble_options))


if __name__ == "__main__":
    unittest.main()
