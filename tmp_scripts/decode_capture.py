#!/usr/bin/env python3
"""Chunked scan of a raw IQ capture for a LoRa frame, avoiding a single huge
FFT over the whole file (impractical at multi-minute captures).

Supports two raw formats:
  int16 -- SDRplay recorder (tmp_scripts/live_lora_capture.py), signed
           16-bit interleaved I/Q, centered at 0.
  uint8 -- RTL-SDR recorder (tmp_scripts/live_lora_capture_rtlsdr.py),
           unsigned 8-bit interleaved I/Q, centered at 127.5.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/src/urh-sdrplay-v3/src"))
import numpy as np  # noqa: E402
from urh.lora import lora_demod as demod  # noqa: E402
from urh.lora.protocols import identify  # noqa: E402


_FORMATS = {
    # dtype, itemsize (bytes per I or Q sample), (i, q) -> normalized complex converter
    "int16": (np.int16, 2, lambda i, q: (i.astype(np.float32) + 1j * q.astype(np.float32)) / 32768.0),
    "uint8": (np.uint8, 1, lambda i, q: ((i.astype(np.float32) - 127.5) + 1j * (q.astype(np.float32) - 127.5)) / 127.5),
}


def scan(path, sf, bw, fs, n_preamble_options=(8, 16), sync_word_options=(0x34, 0x12),
         chunk_sec=3.0, overlap_sec=0.5, sample_format="int16"):
    dtype, item_bytes, to_complex = _FORMATS[sample_format]
    itemsize = 2 * item_bytes  # one I + one Q sample per IQ pair

    chunk_samples = int(chunk_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step = chunk_samples - overlap_samples

    file_size = os.path.getsize(path)
    total_samples = file_size // itemsize

    max_mags = []
    hits = []

    with open(path, "rb") as f:
        offset = 0
        chunk_idx = 0
        while offset < total_samples:
            f.seek(offset * itemsize)
            raw = np.frombuffer(f.read(chunk_samples * itemsize), dtype=dtype)
            if len(raw) < 4:
                break
            iq_pairs = raw.reshape((-1, 2))
            iq = to_complex(iq_pairs[:, 0], iq_pairs[:, 1])
            max_mags.append(float(np.max(np.abs(iq))) if len(iq) else 0.0)

            for result in demod.scan_chunk_for_frames(iq, sf, bw, fs, n_preamble_options, sync_word_options):
                result["chunk_idx"] = chunk_idx
                result["chunk_offset_samples"] = offset
                hits.append(result)
                print(f"[chunk {chunk_idx} @ {offset/fs:.1f}s] DECODED: "
                      f"n_preamble={result['n_preamble']} sync=0x{result['sync_word']:02x} "
                      f"sync_ok={result['sync_ok']} cr={result['cr']} "
                      f"errs={result['uncorrectable_errors']} "
                      f"payload_len={len(result['payload'])} "
                      f"payload_hex={result['payload'].hex()}", flush=True)
                ident = identify.identify(result["payload"], sf=sf, bw=bw)
                print(f"    identify -> {ident}", flush=True)

            if chunk_idx % 10 == 0:
                print(f"  scanned chunk {chunk_idx} (t={offset/fs:.1f}s, max_mag={max_mags[-1]:.4f})", flush=True)

            offset += step
            chunk_idx += 1

    print(f"\n=== scan complete: {chunk_idx} chunks, {len(hits)} decoded frame(s) ===")
    if max_mags:
        print(f"max magnitude overall: {max(max_mags):.4f}, mean of per-chunk maxes: {np.mean(max_mags):.4f}")
    return hits, max_mags


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--sf", type=int, required=True)
    p.add_argument("--bw", type=float, required=True)
    p.add_argument("--fs", type=float, required=True)
    p.add_argument("--format", choices=list(_FORMATS), default="int16")
    args = p.parse_args()
    scan(args.path, args.sf, args.bw, args.fs, sample_format=args.format)
