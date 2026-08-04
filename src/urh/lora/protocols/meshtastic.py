"""Meshtastic protocol parsing on top of a recovered LoRa payload.

Wire format: a 16-byte unencrypted header immediately followed by an
AES-CTR-encrypted `Data` protobuf message. Verified against primary source
(2026-08): meshtastic/firmware `src/mesh/Channels.h` (+.cpp) for the default
channel PSK and its single-byte-index expansion, `src/mesh/CryptoEngine.cpp`
for the AES-CTR nonce construction, and meshtastic/protobufs
`meshtastic/mesh.proto` + `portnums.proto` for the `Data` message layout.

We deliberately do not depend on generated protobuf bindings: `Data` is
small and stable enough that a manual varint/tag scan (see
`parse_data_message`) is simpler than a codegen build step.
"""

import struct

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    HAVE_CRYPTO = True
except ImportError:  # pragma: no cover - exercised via skipUnless in tests
    HAVE_CRYPTO = False

HEADER_LEN = 16

# 16-byte AES-128 key for the public "LongFast"-style primary channel, i.e.
# the well-known single-byte PSK index 1 (base64 "AQ=="). Channels::getKey()
# expands a single-byte PSK by copying `defaultpsk` and bumping its last byte
# by (index - 1); index 1 leaves it unchanged, so this is `defaultpsk` as-is.
# Source: meshtastic/firmware src/mesh/Channels.h.
DEFAULT_PSK = bytes(
    [
        0xD4, 0xF1, 0xBB, 0x3A, 0x20, 0x29, 0x07, 0x59,
        0xF0, 0xBC, 0xFF, 0xAB, 0xCF, 0x4E, 0x69, 0x01,
    ]
)

# meshtastic/protobufs meshtastic/portnums.proto, PortNum enum: values seen
# in practice cluster in 0-79, plus a couple of high private/reserved ranges.
# Used only as a soft plausibility check for protocol identification, not for
# correctness -- an out-of-range portnum just means "less confident", never
# "reject".
_PLAUSIBLE_PORTNUM_MAX = 511


class MeshtasticHeader:
    __slots__ = ("to", "from_", "id", "flags", "channel_hash", "next_hop", "relay_node")

    def __init__(self, to, from_, id_, flags, channel_hash, next_hop, relay_node):
        self.to = to
        self.from_ = from_
        self.id = id_
        self.flags = flags
        self.channel_hash = channel_hash
        self.next_hop = next_hop
        self.relay_node = relay_node

    @property
    def hop_limit(self):
        return self.flags & 0x07

    @property
    def want_ack(self):
        return bool(self.flags & 0x08)

    @property
    def via_mqtt(self):
        return bool(self.flags & 0x10)

    @property
    def hop_start(self):
        return (self.flags >> 5) & 0x07


def parse_header(payload: bytes) -> MeshtasticHeader:
    if len(payload) < HEADER_LEN:
        raise ValueError("payload too short for a Meshtastic header (need >= 16 bytes)")
    to, from_, id_, flags, channel_hash, next_hop, relay_node = struct.unpack_from(
        "<IIIBBBB", payload, 0
    )
    return MeshtasticHeader(to, from_, id_, flags, channel_hash, next_hop, relay_node)


def build_nonce(from_node: int, packet_id: int) -> bytes:
    """16-byte AES-CTR IV = packetId (8 bytes LE, zero-extended from the
    32-bit protobuf field) || fromNode (4 bytes LE) || 4 zero bytes
    (the "extraNonce" slot, only non-zero for Curve25519 DM encryption).
    Source: CryptoEngine::initNonce.
    """
    return struct.pack("<QI4x", packet_id, from_node)


def decrypt_payload(header: MeshtasticHeader, ciphertext: bytes, key: bytes = DEFAULT_PSK) -> bytes:
    if not HAVE_CRYPTO:
        raise RuntimeError(
            "python-cryptography is not installed; install it to decrypt Meshtastic "
            "payloads (e.g. `sudo pacman -S python-cryptography`)."
        )
    nonce = build_nonce(header.from_, header.id)
    decryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


class DataMessage:
    __slots__ = ("portnum", "payload", "want_response", "dest", "source", "request_id", "reply_id", "raw_fields")

    def __init__(self):
        self.portnum = None
        self.payload = b""
        self.want_response = None
        self.dest = None
        self.source = None
        self.request_id = None
        self.reply_id = None
        self.raw_fields = {}


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def parse_data_message(plaintext: bytes) -> DataMessage:
    """Manual protobuf tag/wire-type scan for the `Data` message (see module
    docstring). Fields we recognise: portnum=1 (varint), payload=2 (bytes),
    want_response=3 (varint/bool), dest=4/source=5/request_id=6/reply_id=7
    (all fixed32). Unrecognised field numbers are skipped but kept in
    `raw_fields` for inspection.
    """
    msg = DataMessage()
    pos = 0
    n = len(plaintext)
    while pos < n:
        tag, pos = _read_varint(plaintext, pos)
        field_no = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:  # varint
            value, pos = _read_varint(plaintext, pos)
        elif wire_type == 1:  # 64-bit fixed
            if pos + 8 > n:
                raise ValueError("truncated fixed64 field")
            value = struct.unpack_from("<Q", plaintext, pos)[0]
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(plaintext, pos)
            if length < 0 or pos + length > n:
                raise ValueError("truncated length-delimited field")
            value = plaintext[pos : pos + length]
            pos += length
        elif wire_type == 5:  # 32-bit fixed
            if pos + 4 > n:
                raise ValueError("truncated fixed32 field")
            value = struct.unpack_from("<I", plaintext, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")

        msg.raw_fields[field_no] = value
        if field_no == 1 and wire_type == 0:
            msg.portnum = value
        elif field_no == 2 and wire_type == 2:
            msg.payload = value
        elif field_no == 3 and wire_type == 0:
            msg.want_response = bool(value)
        elif field_no == 4 and wire_type == 5:
            msg.dest = value
        elif field_no == 5 and wire_type == 5:
            msg.source = value
        elif field_no == 6 and wire_type == 5:
            msg.request_id = value
        elif field_no == 7 and wire_type == 5:
            msg.reply_id = value
    return msg


def looks_like_valid_data_message(msg: DataMessage) -> bool:
    """Cheap plausibility check used by the protocol identifier: a
    successfully-parsed-and-decrypted `Data` message with a sane portnum is
    strong corroborating evidence the key/nonce/header alignment was right
    (garbage plaintext almost never parses as a valid tag stream this far).
    """
    return msg.portnum is not None and 0 <= msg.portnum <= _PLAUSIBLE_PORTNUM_MAX


def parse(payload: bytes, key: bytes = DEFAULT_PSK) -> dict:
    """Parse a recovered LoRa payload as a Meshtastic packet.

    Always returns the unencrypted header fields. If `cryptography` is
    available and default-key decryption produces a plausible `Data`
    message, also returns the decoded portnum/payload (and best-effort text
    for TEXT_MESSAGE_APP). Raises ValueError if the payload is too short to
    even contain a header -- callers should treat that as "not Meshtastic".
    """
    header = parse_header(payload)
    ciphertext = payload[HEADER_LEN:]

    result = {
        "to": header.to,
        "from": header.from_,
        "id": header.id,
        "hop_limit": header.hop_limit,
        "want_ack": header.want_ack,
        "via_mqtt": header.via_mqtt,
        "hop_start": header.hop_start,
        "channel_hash": header.channel_hash,
        "next_hop": header.next_hop,
        "relay_node": header.relay_node,
        "decrypted": False,
        "portnum": None,
        "data_payload": None,
        "text": None,
    }

    if not HAVE_CRYPTO:
        return result

    try:
        plaintext = decrypt_payload(header, ciphertext, key)
        data = parse_data_message(plaintext)
    except Exception:
        return result

    if not looks_like_valid_data_message(data):
        return result

    result["decrypted"] = True
    result["portnum"] = data.portnum
    result["data_payload"] = data.payload
    result["dest"] = data.dest
    result["source"] = data.source
    if data.portnum == 1:  # TEXT_MESSAGE_APP
        try:
            result["text"] = data.payload.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return result
