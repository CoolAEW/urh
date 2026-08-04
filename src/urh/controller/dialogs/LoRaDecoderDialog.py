"""Minimal LoRa (chirp spread spectrum) decoder dialog.

Takes the raw IQ of a loaded Signal plus user-supplied SF/BW/sync word and
runs it through the standalone LoRa RX chain in urh.lora, showing the
decoded payload as hex + best-effort ASCII. This is intentionally built
without a .ui file (plain QDialog + code-built layout) to keep it additive
and self-contained -- see LORA_PLAN.md for why LoRa doesn't fit the existing
Signal/Modulator pipeline.

Scanning runs on a QThread with chunk-level progress reporting (via
lora_demod.scan_for_frames), since a signal can be many minutes of IQ and a
single blocking FFT over the whole thing would freeze the UI with no
feedback for a long time.
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
)

from urh.lora.lora_demod import scan_for_frames
from urh.lora.lora_frame_format import DEFAULT_N_PREAMBLE, DEFAULT_SYNC_WORD
from urh.lora.protocols import identify
from urh.signalprocessing.Signal import Signal


class _ScanWorker(QThread):
    progress = pyqtSignal(int, int, float)  # chunk_idx, total_chunks, offset_sec
    finished_ok = pyqtSignal(list, list)  # hits, max_mags
    failed = pyqtSignal(str)

    def __init__(self, iq, sf, bw, fs, n_preamble, sync_word, parent=None):
        super().__init__(parent)
        self.iq = iq
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.n_preamble = n_preamble
        self.sync_word = sync_word
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            hits, max_mags = scan_for_frames(
                self.iq,
                self.sf,
                self.bw,
                self.fs,
                n_preamble_options=(self.n_preamble,),
                sync_word_options=(self.sync_word,),
                progress_cb=lambda i, total, t: self.progress.emit(i, total, t),
                should_stop=lambda: self._stop_requested,
            )
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(hits, max_mags)


class LoRaDecoderDialog(QDialog):
    BANDWIDTHS = [
        ("7.8 kHz", 7800),
        ("10.4 kHz", 10400),
        ("15.6 kHz", 15600),
        ("20.8 kHz", 20800),
        ("31.25 kHz", 31250),
        ("41.7 kHz", 41700),
        ("62.5 kHz", 62500),
        ("125 kHz", 125000),
        ("250 kHz", 250000),
        ("500 kHz", 500000),
    ]

    def __init__(self, signal: Signal, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.worker = None
        self.setWindowTitle(self.tr("LoRa Decoder"))
        self.setMinimumWidth(480)

        self.sf_spinbox = QSpinBox(self)
        self.sf_spinbox.setRange(7, 12)
        self.sf_spinbox.setValue(7)

        self.bw_combobox = QComboBox(self)
        for label, _ in self.BANDWIDTHS:
            self.bw_combobox.addItem(label)
        self.bw_combobox.setCurrentIndex(
            [bw for _, bw in self.BANDWIDTHS].index(125000)
        )

        self.cr_combobox = QComboBox(self)
        for cr in (1, 2, 3, 4):
            self.cr_combobox.addItem(f"4/{4 + cr} (auto-detected from header if left as-is)", cr)
        self.cr_combobox.setCurrentIndex(3)
        self.cr_combobox.setToolTip(
            self.tr("Payload coding rate is read from the frame header automatically; "
                    "this is only informational.")
        )

        self.n_preamble_spinbox = QSpinBox(self)
        self.n_preamble_spinbox.setRange(4, 32)
        self.n_preamble_spinbox.setValue(DEFAULT_N_PREAMBLE)

        self.sync_word_edit = QLineEdit(self)
        self.sync_word_edit.setText(f"0x{DEFAULT_SYNC_WORD:02X}")

        self.sample_rate_label = QLabel(self)
        self._update_sample_rate_label()

        form = QFormLayout()
        form.addRow(self.tr("Spreading factor (SF):"), self.sf_spinbox)
        form.addRow(self.tr("Bandwidth:"), self.bw_combobox)
        form.addRow(self.tr("Coding rate:"), self.cr_combobox)
        form.addRow(self.tr("Preamble length (symbols):"), self.n_preamble_spinbox)
        form.addRow(self.tr("Sync word (hex):"), self.sync_word_edit)
        form.addRow(self.tr("Signal sample rate:"), self.sample_rate_label)

        self.status_label = QLabel(self)
        self.status_label.setText(self.tr("Idle."))

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.result_view = QPlainTextEdit(self)
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText(
            self.tr("Decoded payload will appear here.")
        )
        self.result_view.setFont(self._monospace_font())

        self.button_box = QDialogButtonBox(self)
        self.decode_button = self.button_box.addButton(
            self.tr("Decode"), QDialogButtonBox.ButtonRole.ActionRole
        )
        self.cancel_button = self.button_box.addButton(
            self.tr("Cancel"), QDialogButtonBox.ButtonRole.ActionRole
        )
        self.cancel_button.setEnabled(False)
        self.close_button = self.button_box.addButton(QDialogButtonBox.StandardButton.Close)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.decode_button.clicked.connect(self.on_decode_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.close_button.clicked.connect(self.close)

    @staticmethod
    def _monospace_font():
        from PyQt6.QtGui import QFont

        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        return font

    def _update_sample_rate_label(self):
        rate = self.signal.sample_rate if self.signal is not None else 0
        self.sample_rate_label.setText(f"{rate:,.0f} Hz")

    def _selected_bandwidth(self):
        return self.BANDWIDTHS[self.bw_combobox.currentIndex()][1]

    def _parse_sync_word(self):
        text = self.sync_word_edit.text().strip()
        return int(text, 16) if text.lower().startswith("0x") else int(text, 16)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(5000)
        super().closeEvent(event)

    def _set_running(self, running: bool):
        self.decode_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for w in (self.sf_spinbox, self.bw_combobox, self.cr_combobox,
                  self.n_preamble_spinbox, self.sync_word_edit):
            w.setEnabled(not running)

    def on_cancel_clicked(self):
        if self.worker is not None:
            self.status_label.setText(self.tr("Cancelling..."))
            self.worker.request_stop()

    def on_decode_clicked(self):
        if self.signal is None:
            self.result_view.setPlainText(self.tr("No signal loaded."))
            return

        sf = self.sf_spinbox.value()
        bw = float(self._selected_bandwidth())
        fs = float(self.signal.sample_rate)
        n_preamble = self.n_preamble_spinbox.value()

        try:
            sync_word = self._parse_sync_word()
        except ValueError:
            self.result_view.setPlainText(self.tr("Invalid sync word, expected hex e.g. 0x34"))
            return

        iq = self.signal.iq_array.as_complex64()

        self.result_view.setPlainText("")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(self.tr("Scanning..."))
        self._set_running(True)

        self.worker = _ScanWorker(iq, sf, bw, fs, n_preamble, sync_word, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_scan_finished)
        self.worker.failed.connect(self._on_scan_failed)
        self.worker.finished.connect(lambda: self._set_running(False))
        self.worker.start()

    def _on_progress(self, chunk_idx, total_chunks, offset_sec):
        self.progress_bar.setRange(0, max(1, total_chunks))
        self.progress_bar.setValue(chunk_idx + 1)
        self.status_label.setText(
            self.tr("Scanning chunk {0}/{1} (t={2:.1f}s)...").format(
                chunk_idx + 1, total_chunks, offset_sec
            )
        )

    def _on_scan_failed(self, message):
        self.status_label.setText(self.tr("Error."))
        self.result_view.setPlainText(self.tr("Decode error: ") + message)

    def _on_scan_finished(self, hits, max_mags):
        sf = self.sf_spinbox.value()
        bw = float(self._selected_bandwidth())

        if not hits:
            self.status_label.setText(self.tr("Done -- no frame found."))
            peak = max(max_mags) if max_mags else 0.0
            self.result_view.setPlainText(
                self.tr("No LoRa frame found in this signal.\n\n"
                        "Peak |IQ| magnitude observed: {0:.4f} "
                        "(near/above 1.0 suggests clipping; near 0 suggests "
                        "no signal energy at this frequency/bandwidth).").format(peak)
            )
            return

        # Sort by the composite PHY-level confidence score (see
        # lora_demod.decode_frame) -- this is more reliable than sync_ok/
        # error-count alone, which a degenerate mid-frame collapse into
        # trailing noise can fool (see LORA_PLAN.md).
        hits.sort(key=lambda h: -h["confidence"])

        self.status_label.setText(
            self.tr("Done -- {0} candidate(s) found.").format(len(hits))
        )
        blocks = []
        for h in hits:
            id_result = identify.identify(
                h["payload"], sf=sf, bw=bw,
                decode_confidence=h["confidence"], payload_truncated=h["payload_truncated"],
            )
            truncated_note = (
                f" [TRUNCATED at symbol {h['truncated_at_symbol']} -- signal collapsed mid-payload]"
                if h["payload_truncated"] else ""
            )
            blocks.append(
                f"--- t={h['chunk_offset_samples'] / float(self.signal.sample_rate):.1f}s "
                f"(chunk {h['chunk_idx']}, n_preamble={h['n_preamble']}, "
                f"sync=0x{h['sync_word']:02X}, confidence={h['confidence']:.0%}){truncated_note} ---\n"
                + self._format_result(h, id_result)
            )
        self.result_view.setPlainText("\n\n".join(blocks))

    @staticmethod
    def _format_result(result, id_result):
        payload: bytes = result["payload"]
        hex_str = payload.hex(" ")
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in payload)
        lines = [
            f"Payload length: {len(payload)} bytes",
            f"Coding rate:    4/{4 + result['cr']}",
            f"Sync word OK:   {result['sync_ok']}",
            f"FEC errors:     {result['uncorrectable_errors']}",
            "",
            LoRaDecoderDialog._format_identification(id_result),
            "",
            "Hex:",
            hex_str if hex_str else "(empty)",
            "",
            "ASCII (best-effort):",
            ascii_str if ascii_str else "(empty)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_identification(id_result):
        lines = [
            f"Protocol:       {id_result.protocol} (confidence {id_result.confidence:.0%})",
        ]
        if id_result.protocol == "unknown":
            lines.append("  Not confidently recognized as Meshtastic or MeshCore -- see raw hex/ASCII below.")
        else:
            for key, value in id_result.details.items():
                if isinstance(value, bytes):
                    value = value.hex(" ") if len(value) <= 16 else f"{value[:16].hex(' ')}... ({len(value)} bytes)"
                lines.append(f"  {key}: {value}")
        lines.append("  Reasoning: " + "; ".join(id_result.reasons))
        return "\n".join(lines)
