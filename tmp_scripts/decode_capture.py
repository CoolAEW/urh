#!/usr/bin/env python3
"""Chunked scan of a raw int16 IQ capture for a LoRa frame, avoiding a
single huge FFT over the whole file (impractical at multi-minute captures)."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/src/urh-sdrplay-v3/src"))
import numpy as np  # noqa: E402
from urh.lora import lora_demod as demod  # noqa: E402
from urh.lora.protocols import identify  # noqa: E402


def scan(path, sf, bw, fs, n_preamble_options=(8, 16), sync_word_options=(0x34, 0x12),
         chunk_sec=3.0, overlap_sec=0.5, gain_scale=1.0 / 32768.0):
    chunk_samples = int(chunk_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step = chunk_samples - overlap_samples

    itemsize = 4  # int16 I + int16 Q
    file_size = os.path.getsize(path)
    total_samples = file_size // itemsize

    max_mags = []
    hits = []

    with open(path, "rb") as f:
        offset = 0
        chunk_idx = 0
        while offset < total_samples:
            f.seek(offset * itemsize)
            raw = np.frombuffer(f.read(chunk_samples * itemsize), dtype=np.int16)
            if len(raw) < 4:
                break
            iq_pairs = raw.reshape((-1, 2))
            iq = (iq_pairs[:, 0].astype(np.float32) + 1j * iq_pairs[:, 1].astype(np.float32)) * gain_scale
            max_mags.append(float(np.max(np.abs(iq))) if len(iq) else 0.0)

            for n_preamble in n_preamble_options:
                for sync_word in sync_word_options:
                    try:
                        result = demod.decode_frame(iq, sf, bw, fs=fs, n_preamble=n_preamble, sync_word=sync_word)
                        result["chunk_idx"] = chunk_idx
                        result["chunk_offset_samples"] = offset
                        result["n_preamble"] = n_preamble
                        result["sync_word"] = sync_word
                        hits.append(result)
                        print(f"[chunk {chunk_idx} @ {offset/fs:.1f}s] DECODED: "
                              f"n_preamble={n_preamble} sync=0x{sync_word:02x} "
                              f"sync_ok={result['sync_ok']} cr={result['cr']} "
                              f"errs={result['uncorrectable_errors']} "
                              f"payload_len={len(result['payload'])} "
                              f"payload_hex={result['payload'].hex()}", flush=True)
                        ident = identify.identify(result["payload"], sf=sf, bw=bw)
                        print(f"    identify -> {ident}", flush=True)
                    except demod.LoRaSyncError:
                        pass
                    except Exception as e:
                        print(f"[chunk {chunk_idx}] unexpected error (n_preamble={n_preamble} sync=0x{sync_word:02x}): {e}", flush=True)

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
    args = p.parse_args()
    scan(args.path, args.sf, args.bw, args.fs)
