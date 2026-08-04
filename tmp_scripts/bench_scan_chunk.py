"""Micro-benchmark for Phase 2 dedup, not a committed test."""
import time
import numpy as np
from urh.lora.lora_demod import scan_chunk_for_frames
from urh.lora.lora_modulator import build_frame

rng = np.random.default_rng(1)
sf, bw = 9, 125000.0
frame = build_frame(b"benchmark payload for timing", sf, bw, cr=4)
capture = frame + (rng.standard_normal(len(frame)) + 1j * rng.standard_normal(len(frame))) * 0.05
capture = np.concatenate([capture, (rng.standard_normal(50000) + 1j * rng.standard_normal(50000)) * 0.05])

n_preamble_options = (8, 16)
sync_word_options = (0x34, 0x12)

N = 20
start = time.perf_counter()
for _ in range(N):
    scan_chunk_for_frames(capture, sf, bw, bw, n_preamble_options, sync_word_options)
elapsed = time.perf_counter() - start
print(f"{N} calls in {elapsed:.3f}s ({elapsed/N*1000:.1f}ms/call), "
      f"n_preamble x sync_word = {len(n_preamble_options)}x{len(sync_word_options)}")
