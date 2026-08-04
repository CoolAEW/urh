import struct
import unittest

from urh.lora.protocols import meshcore


def _build_advert(pubkey=None, timestamp=0x11223344, flags=0, lat=None, lon=None,
                   feat1=None, feat2=None, name=None, sign=True):
    if pubkey is None:
        pubkey = bytes(range(32))

    appdata = bytes([flags])
    if flags & meshcore.ADVERT_FLAG_HAS_LOCATION:
        appdata += struct.pack("<ii", int(lat * 1e6), int(lon * 1e6))
    if flags & meshcore.ADVERT_FLAG_HAS_FEATURE1:
        appdata += struct.pack("<H", feat1)
    if flags & meshcore.ADVERT_FLAG_HAS_FEATURE2:
        appdata += struct.pack("<H", feat2)
    if flags & meshcore.ADVERT_FLAG_HAS_NAME:
        appdata += name.encode("utf-8")

    ts_bytes = struct.pack("<I", timestamp)

    if sign and meshcore.HAVE_CRYPTO:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
        pubkey = priv.public_key().public_bytes_raw()
        signature = priv.sign(pubkey + ts_bytes + appdata)
    else:
        signature = bytes(64)

    body = pubkey + ts_bytes + signature + appdata
    header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_ADVERT << 2) | meshcore.ROUTE_TYPE_FLOOD
    path_len_byte = 0  # no path
    return bytes([header_byte, path_len_byte]) + body


class TestHeaderParsing(unittest.TestCase):
    def test_route_and_payload_type_bits(self):
        # route=DIRECT(2)=0b10, payload_type=TXT_MSG(2)=0b0010, ver=0
        b = 0b00_0010_10
        header = meshcore.parse_header_byte(b)
        self.assertEqual(header.route_type, meshcore.ROUTE_TYPE_DIRECT)
        self.assertEqual(header.payload_type, meshcore.PAYLOAD_TYPE_TXT_MSG)
        self.assertEqual(header.payload_ver, 0)

    def test_transport_codes_present_for_transport_routes(self):
        header = meshcore.parse_header_byte(meshcore.ROUTE_TYPE_TRANSPORT_FLOOD)
        self.assertTrue(header.has_transport_codes)
        header = meshcore.parse_header_byte(meshcore.ROUTE_TYPE_FLOOD)
        self.assertFalse(header.has_transport_codes)


class TestAdvertParsing(unittest.TestCase):
    def test_minimal_advert_no_appdata_fields(self):
        payload = _build_advert(flags=0)
        pkt = meshcore.parse(payload)
        self.assertEqual(pkt.header.payload_type, meshcore.PAYLOAD_TYPE_ADVERT)
        self.assertIsNone(pkt.fields["latitude"])
        self.assertIsNone(pkt.fields["name"])

    def test_advert_with_location_and_name(self):
        flags = meshcore.ADVERT_FLAG_HAS_LOCATION | meshcore.ADVERT_FLAG_HAS_NAME
        payload = _build_advert(flags=flags, lat=59.911491, lon=10.757933, name="node-alpha")
        pkt = meshcore.parse(payload)
        self.assertAlmostEqual(pkt.fields["latitude"], 59.911491, places=5)
        self.assertAlmostEqual(pkt.fields["longitude"], 10.757933, places=5)
        self.assertEqual(pkt.fields["name"], "node-alpha")

    def test_advert_with_all_optional_fields(self):
        flags = (
            meshcore.ADVERT_FLAG_HAS_LOCATION
            | meshcore.ADVERT_FLAG_HAS_FEATURE1
            | meshcore.ADVERT_FLAG_HAS_FEATURE2
            | meshcore.ADVERT_FLAG_HAS_NAME
        )
        payload = _build_advert(flags=flags, lat=1.5, lon=-2.5, feat1=7, feat2=99, name="repeater-1")
        pkt = meshcore.parse(payload)
        self.assertAlmostEqual(pkt.fields["latitude"], 1.5, places=5)
        self.assertAlmostEqual(pkt.fields["longitude"], -2.5, places=5)
        self.assertEqual(pkt.fields["feature1"], 7)
        self.assertEqual(pkt.fields["feature2"], 99)
        self.assertEqual(pkt.fields["name"], "repeater-1")

    @unittest.skipUnless(meshcore.HAVE_CRYPTO, "python-cryptography not installed")
    def test_valid_signature_verifies(self):
        payload = _build_advert(flags=meshcore.ADVERT_FLAG_HAS_NAME, name="signed-node", sign=True)
        pkt = meshcore.parse(payload)
        self.assertTrue(pkt.fields["signature_valid"])

    @unittest.skipUnless(meshcore.HAVE_CRYPTO, "python-cryptography not installed")
    def test_tampered_signature_fails(self):
        payload = bytearray(_build_advert(flags=meshcore.ADVERT_FLAG_HAS_NAME, name="signed-node", sign=True))
        payload[36] ^= 0xFF  # flip a byte inside the signature
        pkt = meshcore.parse(bytes(payload))
        self.assertFalse(pkt.fields["signature_valid"])


class TestTextMessageShapedPacket(unittest.TestCase):
    def test_reports_encrypted_key_unknown(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_TXT_MSG << 2) | meshcore.ROUTE_TYPE_FLOOD
        path_len_byte = 0
        dest_hash, src_hash = 0xAB, 0xCD
        mac = b"\x12\x34"
        ciphertext = b"\x00" * 16
        payload = bytes([header_byte, path_len_byte, dest_hash, src_hash]) + mac + ciphertext

        pkt = meshcore.parse(payload)
        self.assertEqual(pkt.fields["dest_hash"], dest_hash)
        self.assertEqual(pkt.fields["src_hash"], src_hash)
        self.assertEqual(pkt.fields["ciphertext"], "encrypted, key unknown")
        self.assertEqual(pkt.fields["ciphertext_len"], len(ciphertext))

    def test_does_not_crash_or_fabricate_plaintext(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_GRP_TXT << 2) | meshcore.ROUTE_TYPE_DIRECT
        payload = bytes([header_byte, 0, 1, 2]) + b"\xff\xff" + b"garbage-ciphertext-bytes"
        pkt = meshcore.parse(payload)
        self.assertNotIn("plaintext", pkt.fields)


class TestPathParsing(unittest.TestCase):
    def test_path_with_1_byte_hashes(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_ADVERT << 2) | meshcore.ROUTE_TYPE_FLOOD
        path = bytes([0xAA, 0xBB, 0xCC])
        path_len_byte = (0b00 << 6) | 3  # hash_size_code=0 (1 byte), count=3
        body = _build_advert()[2:]  # reuse a valid ADVERT body
        payload = bytes([header_byte, path_len_byte]) + path + body
        pkt = meshcore.parse(payload)
        self.assertEqual(pkt.hash_count, 3)
        self.assertEqual(pkt.hash_size, 1)
        self.assertEqual(pkt.path, path)

    def test_reserved_hash_size_code_raises(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_ADVERT << 2) | meshcore.ROUTE_TYPE_FLOOD
        path_len_byte = (0b11 << 6) | 1  # reserved hash size code
        with self.assertRaises(ValueError):
            meshcore.parse(bytes([header_byte, path_len_byte]) + b"\x00" * 101)

    def test_transport_codes_parsed(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_ADVERT << 2) | meshcore.ROUTE_TYPE_TRANSPORT_FLOOD
        transport = struct.pack("<HH", 0x1234, 0x5678)
        path_len_byte = 0
        body = _build_advert()[2:]
        payload = bytes([header_byte]) + transport + bytes([path_len_byte]) + body
        pkt = meshcore.parse(payload)
        self.assertEqual(pkt.transport_code_1, 0x1234)
        self.assertEqual(pkt.transport_code_2, 0x5678)


if __name__ == "__main__":
    unittest.main()
