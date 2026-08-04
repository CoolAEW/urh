"""Minimal LoRa (chirp spread spectrum) decoder dialog.

Takes the raw IQ of a loaded Signal plus user-supplied SF/BW/sync word and
runs it through the standalone LoRa RX chain in urh.lora, showing the
decoded payload as hex + best-effort ASCII. This is intentionally built
without a .ui file (plain QDialog + code-built layout) to keep it additive
and self-contained -- see LORA_PLAN.md for why LoRa doesn't fit the existing
Signal/Modulator pipeline.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)

from urh.lora.lora_demod import decode_frame, LoRaSyncError
from urh.lora.lora_frame_format import DEFAULT_N_PREAMBLE, DEFAULT_SYNC_WORD
from urh.lora.protocols import identify
from urh.signalprocessing.Signal import Signal


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
        self.close_button = self.button_box.addButton(QDialogButtonBox.StandardButton.Close)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.result_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.decode_button.clicked.connect(self.on_decode_clicked)
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

        try:
            result = decode_frame(
                iq, sf, bw, fs=fs, n_preamble=n_preamble, sync_word=sync_word
            )
        except LoRaSyncError as e:
            self.result_view.setPlainText(self.tr("No LoRa frame found: ") + str(e))
            return
        except Exception as e:
            self.result_view.setPlainText(self.tr("Decode error: ") + str(e))
            return

        id_result = identify.identify(result["payload"], sf=sf, bw=bw)
        self.result_view.setPlainText(self._format_result(result, id_result))

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
