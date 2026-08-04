#!/usr/bin/env python3
"""Standalone RTL-SDR IQ recorder for live LoRa capture testing.

Uses the low-level urh.dev.native.lib.rtlsdr Cython binding directly (a
simple synchronous read_sync() loop -- much simpler than SDRplay's
callback-based streaming API).

Writes raw interleaved uint8 I/Q samples (RTL-SDR's native 8-bit unsigned
format, DC around 127.5) to a file -- distinct from the SDRplay recorder's
int16 format, so decode_capture.py needs --format uint8 to read these.

Explicitly disables the RTL2832's digital AGC and sets a fixed manual
tuner gain, since the SDRplay capture pass strongly suggested its AGC was
fighting our manually-requested gain and keeping levels pinned near full
scale regardless of what we asked for.
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/src/urh-sdrplay-v3/src"))
from urh.dev.native.lib import rtlsdr  # noqa: E402


def record(center_freq, duration_sec, out_path, sample_rate=1_024_000,
           gain_tenths_db=125, device_index=0):
    ret = rtlsdr.open(device_index)
    if ret != 0:
        raise RuntimeError(f"open failed: {ret}")

    try:
        print(f"device: {rtlsdr.get_device_list()}", flush=True)

        rtlsdr.set_sample_rate(sample_rate)
        rtlsdr.set_center_freq(int(center_freq))

        # Explicit manual gain, AGC fully off -- both the tuner's own AGC
        # (gain mode) and the RTL2832's digital AGC.
        rtlsdr.set_tuner_gain_mode(1)
        rtlsdr.set_agc_mode(0)
        available_gains = rtlsdr.get_tuner_gains() or []
        chosen_gain = min(available_gains, key=lambda g: abs(g - gain_tenths_db)) \
            if available_gains else gain_tenths_db
        rtlsdr.set_tuner_gain(chosen_gain)

        actual_rate = rtlsdr.get_sample_rate()
        actual_freq = rtlsdr.get_center_freq()
        actual_gain = rtlsdr.get_tuner_gain()
        print(f"stream started: freq={actual_freq} rate={actual_rate} "
              f"gain={actual_gain / 10.0:.1f}dB (agc=off, manual gain mode)", flush=True)

        rtlsdr.reset_buffer()

        read_chunk_samples = 16 * 32 * 512  # rtl-sdr likes multiples of 512
        total_samples_needed = int(duration_sec * actual_rate)
        bytes_written = 0
        start = time.time()
        last_report = start

        with open(out_path, "wb") as f:
            samples_read = 0
            while samples_read < total_samples_needed:
                remaining = total_samples_needed - samples_read
                n = min(read_chunk_samples, remaining)
                data = rtlsdr.read_sync(n)
                f.write(data)
                bytes_written += len(data)
                samples_read += len(data) // 2  # 2 bytes (I, Q) per IQ sample

                now = time.time()
                if now - last_report > 10:
                    print(f"  ...{now - start:.0f}s elapsed, {bytes_written / 1e6:.1f} MB written",
                          flush=True)
                    last_report = now

        print(f"recording done: {bytes_written / 1e6:.1f} MB written to {out_path}", flush=True)
        return bytes_written
    finally:
        try:
            ret = rtlsdr.close()
            print(f"close: {ret}", flush=True)
        except Exception as e:
            print(f"close error: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--rate", type=int, default=1_024_000)
    p.add_argument("--gain", type=int, default=125, help="tenths of a dB, e.g. 125 = 12.5dB")
    args = p.parse_args()
    record(args.freq, args.duration, args.out, sample_rate=args.rate, gain_tenths_db=args.gain)
