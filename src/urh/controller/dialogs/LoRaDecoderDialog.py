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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from urh.lora import lora_autodetect
from urh.lora.lora_demod import scan_for_frames
from urh.lora.lora_frame_format import DEFAULT_N_PREAMBLE, DEFAULT_SYNC_WORD, STANDARD_BANDWIDTHS
from urh.lora.protocols import identify
from urh.signalprocessing.Signal import Signal

#: How much of the loaded signal's IQ (from the start) the "Auto-detect"
#: button scans. A short snippet, not the whole capture -- the point of
#: auto-detect is to identify parameters quickly, and the SF x BW grid
#: sweep is O(60x) a single scan. Long enough to contain a handful of
#: preamble symbols even for a slow (high-SF, low-BW) real-world combo;
#: very-low-BW candidates in the standard grid may still get pruned for
#: being too slow to fit -- see lora_autodetect.estimate.
AUTO_DETECT_SNIPPET_SECONDS = 3.0


class _AutoDetectWorker(QThread):
    finished_ok = pyqtSignal(list)  # list[AutoDetectCandidate]
    failed = pyqtSignal(str)

    def __init__(self, iq_snippet, fs, parent=None):
        super().__init__(parent)
        self.iq_snippet = iq_snippet
        self.fs = fs

    def run(self):
        try:
            candidates = lora_autodetect.estimate(self.iq_snippet, self.fs)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(candidates)


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
    BANDWIDTHS = STANDARD_BANDWIDTHS

    def __init__(self, signal: Signal, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.worker = None
        self.autodetect_worker = None
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

        self.autodetect_button = QPushButton(self.tr("Auto-detect"), self)
        self.autodetect_button.setToolTip(
            self.tr("Scan the first {0:.0f}s of the loaded signal across the standard "
                    "SF/bandwidth grid and fill in the fields above with the best match.")
            .format(AUTO_DETECT_SNIPPET_SECONDS)
        )
        autodetect_row = QHBoxLayout()
        autodetect_row.addWidget(self.autodetect_button)
        autodetect_row.addStretch(1)

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
        layout.addLayout(autodetect_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.decode_button.clicked.connect(self.on_decode_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.close_button.clicked.connect(self.close)
        self.autodetect_button.clicked.connect(self.on_autodetect_clicked)

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
        if self.autodetect_worker is not None and self.autodetect_worker.isRunning():
            self.autodetect_worker.wait(5000)
        super().closeEvent(event)

    def _set_running(self, running: bool):
        self.decode_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.autodetect_button.setEnabled(not running)
        for w in (self.sf_spinbox, self.bw_combobox, self.cr_combobox,
                  self.n_preamble_spinbox, self.sync_word_edit):
            w.setEnabled(not running)

    def on_cancel_clicked(self):
        if self.worker is not None:
            self.status_label.setText(self.tr("Cancelling..."))
            self.worker.request_stop()

    def on_autodetect_clicked(self):
        if self.signal is None:
            self.result_view.setPlainText(self.tr("No signal loaded."))
            return

        fs = float(self.signal.sample_rate)
        full_iq = self.signal.iq_array.as_complex64()
        snippet_len = min(len(full_iq), int(AUTO_DETECT_SNIPPET_SECONDS * fs))
        iq_snippet = full_iq[:snippet_len]

        self.result_view.setPlainText("")
        self.progress_bar.setRange(0, 0)  # indeterminate -- the grid sweep has no natural per-step progress hook
        self.status_label.setText(self.tr("Auto-detecting SF/bandwidth..."))
        self._set_running(True)

        self.autodetect_worker = _AutoDetectWorker(iq_snippet, fs, parent=self)
        self.autodetect_worker.finished_ok.connect(self._on_autodetect_finished)
        self.autodetect_worker.failed.connect(self._on_autodetect_failed)
        self.autodetect_worker.finished.connect(lambda: self._set_running(False))
        self.autodetect_worker.start()

    def _on_autodetect_finished(self, candidates):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)

        if not candidates:
            self.status_label.setText(self.tr("Auto-detect: no candidate found."))
            self.result_view.setPlainText(
                self.tr("No LoRa-like signal found in the first {0:.0f}s of this "
                        "signal at any standard SF/bandwidth combination.")
                .format(AUTO_DETECT_SNIPPET_SECONDS)
            )
            return

        best = candidates[0]
        # Write the winning candidate into the same manually-editable
        # fields a user would set by hand -- mirrors Signal.auto_detect()'s
        # "auto-detect writes into the same fields" pattern for URH's
        # native ASK/FSK/PSK modulations.
        self.sf_spinbox.setValue(best.sf)
        bw_values = [bw for _, bw in self.BANDWIDTHS]
        if best.bw in bw_values:
            self.bw_combobox.setCurrentIndex(bw_values.index(best.bw))
        if best.n_preamble is not None:
            self.n_preamble_spinbox.setValue(best.n_preamble)
        if best.sync_word is not None:
            self.sync_word_edit.setText(f"0x{best.sync_word:02X}")

        confirmed = best.decode_result is not None
        self.status_label.setText(
            self.tr("Auto-detect: SF{0}/{1:,.0f}Hz ({2}) -- fields updated, "
                    "click Decode to run the full scan.")
            .format(best.sf, best.bw, self.tr("confirmed by decode") if confirmed
                    else self.tr("best cheap-stage match, unconfirmed"))
        )

        lines = [
            f"Auto-detect scanned the first {AUTO_DETECT_SNIPPET_SECONDS:.0f}s of the "
            f"signal across the standard SF/bandwidth grid. Top candidates "
            f"(by preamble matched-filter strength):",
            "",
        ]
        for c in candidates[:5]:
            status = "decoded OK" if c.decode_result is not None else "not attempted (outside top candidates)"
            lines.append(f"  SF{c.sf}, {c.bw:,.0f} Hz -- score {c.score:.1f} ({status})")
        self.result_view.setPlainText("\n".join(lines))

    def _on_autodetect_failed(self, message):
        self.status_label.setText(self.tr("Auto-detect error."))
        self.result_view.setPlainText(self.tr("Auto-detect error: ") + message)

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
