"""MeshCore protocol parsing on top of a recovered LoRa payload.

Wire format: `[header(1)][transport_codes(4, optional)][path_len(1)][path]
[payload]`. Verified against primary source (2026-08):
github.com/meshcore-dev/MeshCore `docs/packet_format.md` and
`docs/payloads.md` (v1.12.0 firmware format).

Only ADVERT payloads are fully decodable without secret key material --
they're unencrypted and self-authenticating. TXT_MSG/GRP_TXT/REQ/RESPONSE/
PATH payloads share an outer `[dest_hash][src_hash][MAC][ciphertext]` frame
that we parse structurally, but the ciphertext itself needs a per-node
X25519 shared secret we don't have for arbitrary nearby nodes -- that's
reported as "encrypted, key unknown", not attempted.
"""

import struct

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature

    HAVE_CRYPTO = True
except ImportError:  # pragma: no cover - exercised via skipUnless in tests
    HAVE_CRYPTO = False

ROUTE_TYPE_TRANSPORT_FLOOD = 0x00
ROUTE_TYPE_FLOOD = 0x01
ROUTE_TYPE_DIRECT = 0x02
ROUTE_TYPE_TRANSPORT_DIRECT = 0x03
_ROUTE_TYPES_WITH_TRANSPORT_CODES = (ROUTE_TYPE_TRANSPORT_FLOOD, ROUTE_TYPE_TRANSPORT_DIRECT)

PAYLOAD_TYPE_REQ = 0x00
PAYLOAD_TYPE_RESPONSE = 0x01
PAYLOAD_TYPE_TXT_MSG = 0x02
PAYLOAD_TYPE_ACK = 0x03
PAYLOAD_TYPE_ADVERT = 0x04
PAYLOAD_TYPE_GRP_TXT = 0x05
PAYLOAD_TYPE_GRP_DATA = 0x06
PAYLOAD_TYPE_ANON_REQ = 0x07
PAYLOAD_TYPE_PATH = 0x08
PAYLOAD_TYPE_TRACE = 0x09
PAYLOAD_TYPE_MULTIPART = 0x0A
PAYLOAD_TYPE_CONTROL = 0x0B
PAYLOAD_TYPE_RAW_CUSTOM = 0x0F

PAYLOAD_TYPE_NAMES = {
    PAYLOAD_TYPE_REQ: "REQ",
    PAYLOAD_TYPE_RESPONSE: "RESPONSE",
    PAYLOAD_TYPE_TXT_MSG: "TXT_MSG",
    PAYLOAD_TYPE_ACK: "ACK",
    PAYLOAD_TYPE_ADVERT: "ADVERT",
    PAYLOAD_TYPE_GRP_TXT: "GRP_TXT",
    PAYLOAD_TYPE_GRP_DATA: "GRP_DATA",
    PAYLOAD_TYPE_ANON_REQ: "ANON_REQ",
    PAYLOAD_TYPE_PATH: "PATH",
    PAYLOAD_TYPE_TRACE: "TRACE",
    PAYLOAD_TYPE_MULTIPART: "MULTIPART",
    PAYLOAD_TYPE_CONTROL: "CONTROL",
    PAYLOAD_TYPE_RAW_CUSTOM: "RAW_CUSTOM",
}
# Types that share the [dest_hash][src_hash][MAC][ciphertext] outer frame.
_DEST_SRC_MAC_CIPHERTEXT_TYPES = {
    PAYLOAD_TYPE_REQ,
    PAYLOAD_TYPE_RESPONSE,
    PAYLOAD_TYPE_TXT_MSG,
    PAYLOAD_TYPE_GRP_TXT,
    PAYLOAD_TYPE_PATH,
}

ADVERT_FLAG_HAS_LOCATION = 0x10
ADVERT_FLAG_HAS_FEATURE1 = 0x20
ADVERT_FLAG_HAS_FEATURE2 = 0x40
ADVERT_FLAG_HAS_NAME = 0x80


class MeshCoreHeader:
    __slots__ = ("route_type", "payload_type", "payload_ver")

    def __init__(self, route_type, payload_type, payload_ver):
        self.route_type = route_type
        self.payload_type = payload_type
        self.payload_ver = payload_ver

    @property
    def has_transport_codes(self):
        return self.route_type in _ROUTE_TYPES_WITH_TRANSPORT_CODES

    @property
    def payload_type_name(self):
        return PAYLOAD_TYPE_NAMES.get(self.payload_type, f"reserved(0x{self.payload_type:02X})")


def parse_header_byte(b: int) -> MeshCoreHeader:
    return MeshCoreHeader(route_type=b & 0x03, payload_type=(b >> 2) & 0x0F, payload_ver=(b >> 6) & 0x03)


class ParsedPacket:
    __slots__ = (
        "header", "transport_code_1", "transport_code_2", "hash_count", "hash_size",
        "path", "payload", "fields",
    )

    def __init__(self):
        self.header = None
        self.transport_code_1 = None
        self.transport_code_2 = None
        self.hash_count = 0
        self.hash_size = 1
        self.path = b""
        self.payload = b""
        self.fields = {}


def parse(payload: bytes) -> ParsedPacket:
    """Parse a recovered LoRa payload as a MeshCore packet frame (header,
    optional transport codes, path, payload), then best-effort decode the
    payload body according to its declared type. Raises ValueError if the
    bytes are too short to even contain a header + path_len byte -- callers
    should treat that as "not MeshCore".
    """
    if len(payload) < 2:
        raise ValueError("payload too short for a MeshCore header + path_len")

    pkt = ParsedPacket()
    pos = 0
    header = parse_header_byte(payload[pos])
    pkt.header = header
    pos += 1

    if header.has_transport_codes:
        if pos + 4 > len(payload):
            raise ValueError("truncated transport_codes")
        pkt.transport_code_1, pkt.transport_code_2 = struct.unpack_from("<HH", payload, pos)
        pos += 4

    path_len_byte = payload[pos]
    pos += 1
    hash_count = path_len_byte & 0x3F
    hash_size_code = (path_len_byte >> 6) & 0x03
    if hash_size_code == 0x03:
        raise ValueError("reserved/invalid path hash size code (0b11)")
    hash_size = hash_size_code + 1
    pkt.hash_count = hash_count
    pkt.hash_size = hash_size

    path_bytes = hash_count * hash_size
    if pos + path_bytes > len(payload):
        raise ValueError("truncated path")
    pkt.path = payload[pos : pos + path_bytes]
    pos += path_bytes

    pkt.payload = payload[pos:]
    pkt.fields = _decode_payload_body(header.payload_type, pkt.payload)
    return pkt


def _decode_payload_body(payload_type: int, body: bytes) -> dict:
    if payload_type == PAYLOAD_TYPE_ADVERT:
        return _decode_advert(body)
    if payload_type == PAYLOAD_TYPE_ACK:
        return _decode_ack(body)
    if payload_type in _DEST_SRC_MAC_CIPHERTEXT_TYPES:
        return _decode_dest_src_mac_ciphertext(body)
    return {"note": "payload type not decoded in detail, raw bytes only"}


def _decode_advert(body: bytes) -> dict:
    if len(body) < 100:
        raise ValueError("ADVERT payload shorter than fixed pubkey+timestamp+signature (100 bytes)")
    pubkey = body[0:32]
    (timestamp,) = struct.unpack_from("<I", body, 32)
    signature = body[36:100]
    appdata = body[100:]

    result = {
        "public_key": pubkey,
        "timestamp": timestamp,
        "signature": signature,
        "latitude": None,
        "longitude": None,
        "feature1": None,
        "feature2": None,
        "name": None,
        "signature_valid": None,
    }

    if appdata:
        flags = appdata[0]
        p = 1
        if flags & ADVERT_FLAG_HAS_LOCATION:
            if p + 8 > len(appdata):
                raise ValueError("truncated ADVERT appdata (lat/lon)")
            lat_raw, lon_raw = struct.unpack_from("<ii", appdata, p)
            result["latitude"] = lat_raw / 1e6
            result["longitude"] = lon_raw / 1e6
            p += 8
        if flags & ADVERT_FLAG_HAS_FEATURE1:
            if p + 2 > len(appdata):
                raise ValueError("truncated ADVERT appdata (feature1)")
            result["feature1"] = struct.unpack_from("<H", appdata, p)[0]
            p += 2
        if flags & ADVERT_FLAG_HAS_FEATURE2:
            if p + 2 > len(appdata):
                raise ValueError("truncated ADVERT appdata (feature2)")
            result["feature2"] = struct.unpack_from("<H", appdata, p)[0]
            p += 2
        if flags & ADVERT_FLAG_HAS_NAME:
            result["name"] = appdata[p:].decode("utf-8", errors="replace")
        result["flags"] = flags

    if HAVE_CRYPTO:
        try:
            Ed25519PublicKey.from_public_bytes(pubkey).verify(signature, pubkey + body[32:36] + appdata)
            result["signature_valid"] = True
        except InvalidSignature:
            result["signature_valid"] = False
        except Exception:
            result["signature_valid"] = None

    return result


def _decode_ack(body: bytes) -> dict:
    if len(body) < 4:
        raise ValueError("ACK payload shorter than 4-byte checksum")
    (checksum,) = struct.unpack_from("<I", body, 0)
    return {"checksum": checksum}


def _decode_dest_src_mac_ciphertext(body: bytes) -> dict:
    if len(body) < 4:
        raise ValueError("payload shorter than dest_hash+src_hash+MAC (4 bytes)")
    dest_hash, src_hash = body[0], body[1]
    mac = body[2:4]
    ciphertext = body[4:]
    return {
        "dest_hash": dest_hash,
        "src_hash": src_hash,
        "mac": mac,
        "ciphertext_len": len(ciphertext),
        "ciphertext": "encrypted, key unknown",
    }
