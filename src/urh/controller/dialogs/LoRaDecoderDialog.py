"""Minimal LoRa (chirp spread spectrum) decoder dialog.

Takes the raw IQ of a loaded Signal plus user-supplied SF/BW/sync word and
runs it through the standalone LoRa RX chain in urh.lora, showing decoded
candidates in a table (sortable by confidence, one row per scan hit) with a
detail pane for the full hex/ASCII/identification breakdown of whichever row
is selected. This is intentionally built without a .ui file (plain QDialog +
code-built layout) to keep it additive and self-contained -- see
LORA_PLAN.md for why LoRa doesn't fit the existing Signal/Modulator
pipeline as a first-class modulation_type.

Scanning runs on a QThread with chunk-level progress reporting (via
lora_demod.scan_for_frames), since a signal can be many minutes of IQ and a
single blocking FFT over the whole thing would freeze the UI with no
feedback for a long time.
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
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

_HITS_TABLE_COLUMNS = (
    "Time", "Confidence", "Sync OK", "Protocol", "CR", "FEC errors", "Truncated", "Payload preview",
)


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
        # Parallel lists: row i of hits_table corresponds to
        # self._hits[i] / self._id_results[i]. Populated by _on_scan_finished,
        # read back by _on_hits_table_selection_changed.
        self._hits = []
        self._id_results = []
        self.setWindowTitle(self.tr("LoRa Decoder"))
        self.setMinimumWidth(640)
        self.resize(760, 560)

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

        # One row per scan hit, sorted by confidence descending -- see
        # _on_scan_finished. Styling mirrors ui_csv_wizard.py's
        # tableWidgetPreview, the closest existing lightweight QTableWidget
        # reference in this codebase (URH has no global theme/.qss file).
        self.hits_table = QTableWidget(0, len(_HITS_TABLE_COLUMNS), self)
        self.hits_table.setHorizontalHeaderLabels(_HITS_TABLE_COLUMNS)
        self.hits_table.setAlternatingRowColors(True)
        self.hits_table.horizontalHeader().setCascadingSectionResizes(False)
        self.hits_table.horizontalHeader().setStretchLastSection(True)
        self.hits_table.verticalHeader().setVisible(False)
        self.hits_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.hits_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.hits_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hits_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        self.detail_view = QPlainTextEdit(self)
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText(
            self.tr("Select a row above to see the full decoded payload here, "
                    "or run Decode/Auto-detect first.")
        )
        self.detail_view.setFont(self._monospace_font())

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
        layout.addWidget(self.hits_table, stretch=1)
        layout.addWidget(self.detail_view, stretch=1)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.decode_button.clicked.connect(self.on_decode_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.close_button.clicked.connect(self.close)
        self.autodetect_button.clicked.connect(self.on_autodetect_clicked)
        self.hits_table.itemSelectionChanged.connect(self._on_hits_table_selection_changed)

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

    def _clear_results(self, placeholder_text=""):
        self._hits = []
        self._id_results = []
        self.hits_table.setRowCount(0)
        self.detail_view.setPlainText(placeholder_text)

    def on_cancel_clicked(self):
        if self.worker is not None:
            self.status_label.setText(self.tr("Cancelling..."))
            self.worker.request_stop()

    def on_autodetect_clicked(self):
        if self.signal is None:
            self._clear_results(self.tr("No signal loaded."))
            return

        fs = float(self.signal.sample_rate)
        full_iq = self.signal.iq_array.as_complex64()
        snippet_len = min(len(full_iq), int(AUTO_DETECT_SNIPPET_SECONDS * fs))
        iq_snippet = full_iq[:snippet_len]

        self._clear_results()
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
            self.detail_view.setPlainText(
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
        self.detail_view.setPlainText("\n".join(lines))

    def _on_autodetect_failed(self, message):
        self.status_label.setText(self.tr("Auto-detect error."))
        self.detail_view.setPlainText(self.tr("Auto-detect error: ") + message)

    def on_decode_clicked(self):
        if self.signal is None:
            self._clear_results(self.tr("No signal loaded."))
            return

        sf = self.sf_spinbox.value()
        bw = float(self._selected_bandwidth())
        fs = float(self.signal.sample_rate)
        n_preamble = self.n_preamble_spinbox.value()

        try:
            sync_word = self._parse_sync_word()
        except ValueError:
            self._clear_results(self.tr("Invalid sync word, expected hex e.g. 0x34"))
            return

        iq = self.signal.iq_array.as_complex64()

        self._clear_results()
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
        self.detail_view.setPlainText(self.tr("Decode error: ") + message)

    def _on_scan_finished(self, hits, max_mags):
        sf = self.sf_spinbox.value()
        bw = float(self._selected_bandwidth())

        if not hits:
            self.status_label.setText(self.tr("Done -- no frame found."))
            peak = max(max_mags) if max_mags else 0.0
            self.detail_view.setPlainText(
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

        self._hits = hits
        self._id_results = [
            identify.identify(
                h["payload"], sf=sf, bw=bw,
                decode_confidence=h["confidence"], payload_truncated=h["payload_truncated"],
            )
            for h in hits
        ]
        self._populate_hits_table()
        if hits:
            self.hits_table.selectRow(0)

    def _populate_hits_table(self):
        self.hits_table.setRowCount(len(self._hits))
        for row, (h, id_result) in enumerate(zip(self._hits, self._id_results)):
            time_s = h["chunk_offset_samples"] / float(self.signal.sample_rate)
            payload: bytes = h["payload"]
            preview = payload[:8].hex(" ")
            if len(payload) > 8:
                preview += "..."

            values = [
                f"{time_s:.1f}s",
                f"{h['confidence']:.0%}",
                "yes" if h["sync_ok"] else "no",
                id_result.protocol,
                f"4/{4 + h['cr']}",
                str(h["uncorrectable_errors"]),
                f"symbol {h['truncated_at_symbol']}" if h["payload_truncated"] else "",
                preview if preview else "(empty)",
            ]
            for col, value in enumerate(values):
                self.hits_table.setItem(row, col, QTableWidgetItem(value))

    def _on_hits_table_selection_changed(self):
        rows = self.hits_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row >= len(self._hits):
            return
        self.detail_view.setPlainText(
            self._format_result(self._hits[row], self._id_results[row])
        )

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
            f"Confidence:     {result['confidence']:.0%}",
        ]
        if result["payload_truncated"]:
            lines.append(
                f"Truncated:      yes, at payload symbol {result['truncated_at_symbol']} "
                f"(signal collapsed mid-payload -- see LORA_PLAN.md)"
            )
        lines += [
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
