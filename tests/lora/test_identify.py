import struct
import unittest

from urh.lora.protocols import identify, meshcore, meshtastic
from tests.lora.test_meshtastic import _build_data_protobuf, _build_packet
from tests.lora.test_meshcore import _build_advert


class TestIdentifyMeshtastic(unittest.TestCase):
    def test_decrypting_packet_is_confidently_meshtastic(self):
        protobuf = _build_data_protobuf(portnum=1, payload=b"hi")
        payload = _build_packet(to_=1, from_=2, id_=3, plaintext=protobuf)

        result = identify.identify(payload, sf=11, bw=250000)
        self.assertEqual(result.protocol, "meshtastic")
        self.assertGreater(result.confidence, 0.9)
        self.assertTrue(result.details["decrypted"])

    def test_decrypting_packet_without_sf_bw_still_identified(self):
        protobuf = _build_data_protobuf(portnum=1, payload=b"hi")
        payload = _build_packet(to_=1, from_=2, id_=3, plaintext=protobuf)

        result = identify.identify(payload)  # no SF/BW given
        self.assertEqual(result.protocol, "meshtastic")

    def test_header_only_no_decrypt_no_sf_bw_is_not_confidently_wrong(self):
        # header-shaped but ciphertext is garbage under the default key. The
        # resulting high-entropy bytes can *also* structurally pass as a
        # MeshCore header (its 1-byte header is a much smaller, weaker
        # fingerprint -- see plan notes) -- so this is inherently ambiguous.
        # What we actually require is "no confident wrong answer".
        payload = _build_packet(to_=1, from_=2, id_=3, plaintext=b"", key=bytes(range(16)))
        payload = payload + b"\x99" * 8  # non-decrypting ciphertext tail

        result = identify.identify(payload)
        if result.protocol != "unknown":
            self.assertLess(result.confidence, 0.6)

    def test_sf_bw_preset_match_alone_is_corroborating(self):
        # 16-byte header, empty ciphertext -> never decrypts to a valid Data
        # message (no fields at all), but SF/BW matches LONG_FAST.
        payload = struct.pack("<IIIBBBB", 0, 0, 0, 0, 0, 0, 0)
        result = identify.identify(payload, sf=11, bw=250000)
        self.assertEqual(result.protocol, "meshtastic")
        self.assertLess(result.confidence, 0.7)  # weaker than a real decrypt


class TestIdentifyMeshCore(unittest.TestCase):
    def test_advert_at_default_sf_bw_is_confidently_meshcore(self):
        payload = _build_advert(flags=meshcore.ADVERT_FLAG_HAS_NAME, name="node-x", sign=True)
        result = identify.identify(payload, sf=8, bw=62500)
        self.assertEqual(result.protocol, "meshcore")
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(result.details["name"], "node-x")

    def test_txt_msg_shaped_without_sf_bw_is_weak_meshcore_or_unknown(self):
        header_byte = (0 << 6) | (meshcore.PAYLOAD_TYPE_TXT_MSG << 2) | meshcore.ROUTE_TYPE_FLOOD
        payload = bytes([header_byte, 0, 0xAB, 0xCD]) + b"\x11\x22" + b"\x00" * 16
        result = identify.identify(payload)
        # Weak structural-only signal: either classified with low confidence,
        # or below the reporting threshold entirely -- both are acceptable,
        # a confident wrong answer is not.
        if result.protocol != "unknown":
            self.assertEqual(result.protocol, "meshcore")
            self.assertLess(result.confidence, 0.6)


class TestIdentifyUnknown(unittest.TestCase):
    def test_too_short_for_either_protocol_is_unknown(self):
        result = identify.identify(b"\x00")
        self.assertEqual(result.protocol, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_does_not_crash_on_random_bytes(self):
        import random

        rng = random.Random(1234)
        for _ in range(50):
            n = rng.randint(0, 60)
            payload = bytes(rng.randrange(256) for _ in range(n))
            result = identify.identify(payload, sf=rng.choice([7, 8, 9, 10, 11, 12]),
                                        bw=rng.choice([62500, 125000, 250000, 500000]))
            self.assertIn(result.protocol, ("meshtastic", "meshcore", "unknown"))
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
