#!/usr/bin/env python3
"""Launch URH with the latest MeshCore capture already loaded and the
LoRa Decoder dialog pre-filled and scanned, so it's immediately visible."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/src/urh-sdrplay-v3/src"))

import numpy as np
from PyQt6.QtWidgets import QApplication

from urh.controller.MainController import MainController
from urh.controller.dialogs.LoRaDecoderDialog import LoRaDecoderDialog
from urh.signalprocessing.IQArray import IQArray
from urh.signalprocessing.Signal import Signal

app = QApplication(sys.argv)

path = os.path.expanduser(
    "~/src/urh-sdrplay-v3/tmp_scripts/captures/meshcore_rtlsdr_long1.iq"
)
fs = 1_000_000.0
# Load only a window around the best hit (t=167.5s), not the full 600MB/
# 5-minute file -- loading and converting the whole thing pushed this
# 7.5GB-RAM machine into heavy swapping and an OOM kill last time.
window_start_sec, window_len_sec = 150.0, 40.0
itemsize = 2  # uint8 I + uint8 Q
with open(path, "rb") as f:
    f.seek(int(window_start_sec * fs) * itemsize)
    raw = np.frombuffer(f.read(int(window_len_sec * fs) * itemsize), dtype=np.uint8)
iq_pairs = raw.reshape((-1, 2))
iq = (
    (iq_pairs[:, 0].astype(np.float32) - 127.5)
    + 1j * (iq_pairs[:, 1].astype(np.float32) - 127.5)
) / 127.5

signal = Signal(path, "meshcore_rtlsdr_long1 (869.618 MHz, 5 min)")
signal.iq_array = IQArray(iq.astype(np.complex64))
signal.sample_rate = fs

main_window = MainController()
main_window.show()
# Deliberately not calling main_window.add_signal(signal) here: that runs
# URH's native auto-detect-modulation pipeline (wavelet analysis over the
# whole raw IQ array) plus a full-waveform render, both sized for typical
# short native captures -- on this 5-minute/300M-sample raw LoRa capture it
# OOM-killed the process. The LoRa dialog doesn't need the signal to be a
# native tab at all; it reads iq_array directly.

dlg = LoRaDecoderDialog(signal, parent=main_window)
# Exercise the new preset dropdown through its real UI path (not manual
# field-setting) -- selecting "MeshCore - EU868 Default" should fill in
# SF=8, BW=62.5kHz, n_preamble=16, sync=0x12 all at once.
meshcore_preset_index = [
    i for i in range(dlg.preset_combobox.count())
    if "MeshCore" in dlg.preset_combobox.itemText(i)
][0]
dlg.preset_combobox.setCurrentIndex(meshcore_preset_index)
dlg.show()
dlg.on_decode_clicked()

sys.exit(app.exec())
