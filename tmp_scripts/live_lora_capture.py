#!/usr/bin/env python3
"""Standalone SDRplay RSPdx IQ recorder for live LoRa capture testing.

Uses the low-level urh.dev.native.lib.sdrplay Cython binding directly
(a multiprocessing.connection-style object with .send_bytes() is what the
native stream callback expects -- a plain mp.Pipe() end satisfies that
without needing URH's full subprocess Device machinery).

Writes raw interleaved int16 I/Q samples to a file, matching
SDRPlay.bytes_to_iq's format (np.frombuffer(..., dtype=np.int16).reshape((-1, 2))).
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/src/urh-sdrplay-v3/src"))
from urh.dev.native.lib import sdrplay  # noqa: E402


def record(center_freq, duration_sec, out_path, sample_rate=2_000_000.0,
           bandwidth=300_000.0, gain=45, if_mode=0.0):
    parent_conn, child_conn = mp.Pipe(duplex=False)

    ret = sdrplay.open_api()
    if ret != 0:
        raise RuntimeError(f"open_api failed: {ret}")

    try:
        devices = sdrplay.get_devices()
        if not devices:
            raise RuntimeError("no SDRplay devices found")
        print(f"devices: {devices}", flush=True)

        ret = sdrplay.set_device_index(0)
        if ret != 0:
            raise RuntimeError(f"set_device_index failed: {ret}")
        sdrplay.set_gr_mode_for_dev_model(devices[0]["hw_version"])

        bytes_written = 0
        with open(out_path, "wb") as f:
            ret = sdrplay.init_stream(gain, sample_rate, center_freq, bandwidth, if_mode, child_conn)
            if ret != 0:
                raise RuntimeError(f"init_stream failed: {ret}")
            print(f"stream started: freq={center_freq} rate={sample_rate} bw={bandwidth} gain={gain}", flush=True)

            start = time.time()
            end_time = start + duration_sec
            last_report = start
            while time.time() < end_time:
                if parent_conn.poll(0.5):
                    data = parent_conn.recv_bytes()
                    f.write(data)
                    bytes_written += len(data)
                now = time.time()
                if now - last_report > 10:
                    print(f"  ...{now - start:.0f}s elapsed, {bytes_written / 1e6:.1f} MB written", flush=True)
                    last_report = now

            # drain anything still queued
            drain_deadline = time.time() + 1.0
            while time.time() < drain_deadline and parent_conn.poll(0.2):
                data = parent_conn.recv_bytes()
                f.write(data)
                bytes_written += len(data)

        print(f"recording done: {bytes_written / 1e6:.1f} MB written to {out_path}", flush=True)
        return bytes_written
    finally:
        try:
            ret = sdrplay.close_stream()
            print(f"close_stream: {ret}", flush=True)
        except Exception as e:
            print(f"close_stream error: {e}", flush=True)
        time.sleep(2.5)  # SDRplay API needs ~2-3s to cleanly release, per prior notes
        try:
            ret = sdrplay.release_device_index()
            print(f"release_device_index: {ret}", flush=True)
        except Exception as e:
            print(f"release_device_index error: {e}", flush=True)
        try:
            ret = sdrplay.close_api()
            print(f"close_api: {ret}", flush=True)
        except Exception as e:
            print(f"close_api error: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--rate", type=float, default=2_000_000.0)
    p.add_argument("--bw", type=float, default=300_000.0)
    p.add_argument("--gain", type=int, default=45)
    args = p.parse_args()
    record(args.freq, args.duration, args.out, sample_rate=args.rate, bandwidth=args.bw, gain=args.gain)
