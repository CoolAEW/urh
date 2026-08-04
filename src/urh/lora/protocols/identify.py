"""Best-effort protocol identification for a recovered LoRa payload.

Neither Meshtastic nor MeshCore packets carry an explicit "I am protocol X"
marker, so identification combines structural plausibility of the payload
bytes with the SF/BW the frame was demodulated with (both projects mostly
stick to a small set of presets) and, where possible, whether the payload
actually decodes/decrypts to something sane -- that last signal is the
strongest one we have, since garbage essentially never produces a valid
protobuf tag stream or a length-consistent MeshCore frame by chance.

This is inherently fuzzy: report a confidence score, not a boolean.
"""

from urh.lora.protocols import meshcore, meshtastic

# (sf, bw_hz) -> ModemPreset name, from meshtastic/protobufs config.proto.
MESHTASTIC_PRESETS = {
    (7, 500000): "SHORT_TURBO",
    (7, 250000): "SHORT_FAST",
    (8, 250000): "SHORT_SLOW",
    (9, 250000): "MEDIUM_FAST",
    (10, 250000): "MEDIUM_SLOW",
    (11, 500000): "LONG_TURBO",
    (11, 250000): "LONG_FAST",
    (11, 125000): "LONG_MODERATE",
    (12, 125000): "LONG_SLOW",
}

# EU868 default per MeshCore firmware.
MESHCORE_DEFAULT_SF_BW = (8, 62500)

# A bare structural parse alone (no SF/BW corroboration, no successful
# decode) isn't enough to call it identified -- below this, report "unknown"
# even though one of the per-protocol scores may be nonzero.
_MIN_CONFIDENCE = 0.35

# Header/payload_type values MeshCore firmware v1.12.0 never emits.
_MESHCORE_RESERVED_PAYLOAD_TYPES = {0x0C, 0x0D, 0x0E}


class IdentificationResult:
    __slots__ = ("protocol", "confidence", "details", "reasons")

    def __init__(self, protocol, confidence, details, reasons):
        self.protocol = protocol
        self.confidence = confidence
        self.details = details
        self.reasons = reasons

    def __repr__(self):
        return f"IdentificationResult({self.protocol!r}, confidence={self.confidence:.2f})"


def _score_meshtastic(payload, sf, bw):
    try:
        meshtastic.parse_header(payload)
    except ValueError as e:
        return 0.0, {}, [f"not Meshtastic: {e}"]

    score = 0.3
    reasons = ["16-byte header parses"]

    preset = MESHTASTIC_PRESETS.get((sf, bw)) if sf is not None and bw is not None else None
    if preset is not None:
        score += 0.3
        reasons.append(f"SF/BW matches Meshtastic preset {preset}")

    details = meshtastic.parse(payload)
    if details.get("decrypted"):
        score += 0.4
        reasons.append("decrypts to a plausible Data protobuf with the default channel key")
    elif not meshtastic.HAVE_CRYPTO:
        reasons.append("python-cryptography not installed; could not attempt decryption")

    return score, details, reasons


def _score_meshcore(payload, sf, bw):
    try:
        pkt = meshcore.parse(payload)
    except ValueError as e:
        return 0.0, {}, [f"not MeshCore: {e}"]

    if pkt.header.payload_type in _MESHCORE_RESERVED_PAYLOAD_TYPES:
        return 0.0, {}, ["payload_type is a reserved value MeshCore firmware never emits"]

    score = 0.3
    reasons = ["1-byte header + path parses structurally"]

    if pkt.header.payload_ver == 0:
        score += 0.15
        reasons.append("payload_ver is 0 (the only version in current use)")

    if sf is not None and bw is not None and (sf, bw) == MESHCORE_DEFAULT_SF_BW:
        score += 0.3
        reasons.append("SF/BW matches MeshCore's EU868 default (SF8/BW62.5kHz)")

    details = {
        "route_type": pkt.header.route_type,
        "payload_type": pkt.header.payload_type_name,
        "payload_ver": pkt.header.payload_ver,
        "hash_count": pkt.hash_count,
        "hash_size": pkt.hash_size,
        **pkt.fields,
    }

    if pkt.header.payload_type == meshcore.PAYLOAD_TYPE_ADVERT and "public_key" in pkt.fields:
        score += 0.25
        reasons.append("ADVERT payload fully parsed (self-authenticating structure)")
        if pkt.fields.get("signature_valid") is True:
            score += 0.2
            reasons.append("Ed25519 signature verifies")
        elif pkt.fields.get("signature_valid") is False:
            score -= 0.2
            reasons.append("Ed25519 signature does NOT verify (weakens confidence)")

    return score, details, reasons


def identify(
    payload: bytes,
    sf: int = None,
    bw: float = None,
    decode_confidence: float = None,
    payload_truncated: bool = False,
) -> IdentificationResult:
    """Identify a recovered LoRa payload as Meshtastic, MeshCore, or
    unknown. `sf`/`bw` (the demod parameters used to recover this payload)
    are optional but substantially improve confidence when supplied.

    `decode_confidence` (the LoRa-PHY-level decode_frame confidence score,
    0-1) and `payload_truncated` are optional signals from the PHY layer
    about how much to trust `payload` in the first place -- without them, a
    truncated/low-confidence payload that still happens to structurally
    parse (e.g. a Meshtastic 16-byte header surviving a mid-payload
    collapse) could score as high as a fully clean decode. When supplied,
    they scale the final score down; they never *raise* it, since a
    structural match on truncated/uncertain bytes is never more trustworthy
    than the same match on a fully confident decode.
    """
    mt_score, mt_details, mt_reasons = _score_meshtastic(payload, sf, bw)
    mc_score, mc_details, mc_reasons = _score_meshcore(payload, sf, bw)

    if payload_truncated:
        mt_score *= 0.5
        mc_score *= 0.5
        mt_reasons = mt_reasons + ["payload was truncated (mid-frame signal collapse) -- reduced confidence"]
        mc_reasons = mc_reasons + ["payload was truncated (mid-frame signal collapse) -- reduced confidence"]
    if decode_confidence is not None:
        mt_score *= decode_confidence
        mc_score *= decode_confidence

    best_score = max(mt_score, mc_score)
    if best_score < _MIN_CONFIDENCE:
        reasons = ["no confident structural match for either protocol"]
        if mt_score > 0:
            reasons += [f"(Meshtastic, weak: {r})" for r in mt_reasons]
        if mc_score > 0:
            reasons += [f"(MeshCore, weak: {r})" for r in mc_reasons]
        return IdentificationResult("unknown", best_score, {}, reasons)

    if mt_score >= mc_score:
        return IdentificationResult("meshtastic", min(mt_score, 1.0), mt_details, mt_reasons)
    return IdentificationResult("meshcore", min(mc_score, 1.0), mc_details, mc_reasons)
