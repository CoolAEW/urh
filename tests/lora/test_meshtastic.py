import struct
import unittest

from urh.lora.protocols import meshtastic


def _build_data_protobuf(portnum: int, payload: bytes) -> bytes:
    """Minimal manual protobuf encode of a `Data` message with just
    portnum (field 1, varint) and payload (field 2, bytes) set -- mirrors
    what real Meshtastic firmware would produce for a simple text message.
    """
    assert portnum < 128, "test helper only supports single-byte varints"
    return bytes([0x08, portnum, 0x12, len(payload)]) + payload


def _build_packet(to_, from_, id_, hop_limit=3, want_ack=False, via_mqtt=False,
                   hop_start=3, channel_hash=0x08, next_hop=0, relay_node=0,
                   plaintext=b"", key=meshtastic.DEFAULT_PSK):
    flags = (hop_limit & 0x07) | ((1 if want_ack else 0) << 3) | ((1 if via_mqtt else 0) << 4) | ((hop_start & 0x07) << 5)
    header = struct.pack("<IIIBBBB", to_, from_, id_, flags, channel_hash, next_hop, relay_node)
    nonce = meshtastic.build_nonce(from_, id_)
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    encryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return header + ciphertext


class TestHeaderParsing(unittest.TestCase):
    def test_round_trip_fields(self):
        payload = _build_packet(
            to_=0xFFFFFFFF, from_=0x12345678, id_=0xAABBCCDD,
            hop_limit=5, want_ack=True, via_mqtt=True, hop_start=7,
            channel_hash=0x2A, next_hop=0x11, relay_node=0x22,
            plaintext=b"",
        )
        header = meshtastic.parse_header(payload)
        self.assertEqual(header.to, 0xFFFFFFFF)
        self.assertEqual(header.from_, 0x12345678)
        self.assertEqual(header.id, 0xAABBCCDD)
        self.assertEqual(header.hop_limit, 5)
        self.assertTrue(header.want_ack)
        self.assertTrue(header.via_mqtt)
        self.assertEqual(header.hop_start, 7)
        self.assertEqual(header.channel_hash, 0x2A)
        self.assertEqual(header.next_hop, 0x11)
        self.assertEqual(header.relay_node, 0x22)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            meshtastic.parse_header(b"\x00" * 10)


class TestDataMessageParsing(unittest.TestCase):
    def test_round_trip_simple_fields(self):
        protobuf = _build_data_protobuf(portnum=1, payload=b"hello")
        msg = meshtastic.parse_data_message(protobuf)
        self.assertEqual(msg.portnum, 1)
        self.assertEqual(msg.payload, b"hello")

    def test_fixed32_fields(self):
        # field 4 (dest, fixed32) tag = (4<<3)|5 = 0x25
        protobuf = bytes([0x25]) + struct.pack("<I", 0xDEADBEEF)
        msg = meshtastic.parse_data_message(protobuf)
        self.assertEqual(msg.dest, 0xDEADBEEF)


@unittest.skipUnless(meshtastic.HAVE_CRYPTO, "python-cryptography not installed")
class TestFullPacketDecrypt(unittest.TestCase):
    def test_text_message_round_trip(self):
        protobuf = _build_data_protobuf(portnum=1, payload=b"hello mesh")
        payload = _build_packet(to_=0xFFFFFFFF, from_=0x445566AA, id_=0x01020304, plaintext=protobuf)

        result = meshtastic.parse(payload)
        self.assertTrue(result["decrypted"])
        self.assertEqual(result["portnum"], 1)
        self.assertEqual(result["text"], "hello mesh")
        self.assertEqual(result["from"], 0x445566AA)

    def test_wrong_key_does_not_falsely_decrypt(self):
        protobuf = _build_data_protobuf(portnum=1, payload=b"secret")
        wrong_key = bytes(range(16))
        payload = _build_packet(to_=1, from_=2, id_=3, plaintext=protobuf, key=wrong_key)

        result = meshtastic.parse(payload)
        # Decrypting with the wrong key almost never happens to produce a
        # syntactically valid Data protobuf by chance.
        self.assertFalse(result["decrypted"])

    def test_nonce_construction_matches_spec(self):
        # packetId (8B LE, zero-extended from 32-bit) || fromNode (4B LE) || 4 zero bytes
        nonce = meshtastic.build_nonce(from_node=0x11223344, packet_id=0x99887766)
        self.assertEqual(len(nonce), 16)
        self.assertEqual(nonce[:8], struct.pack("<Q", 0x99887766))
        self.assertEqual(nonce[8:12], struct.pack("<I", 0x11223344))
        self.assertEqual(nonce[12:16], b"\x00\x00\x00\x00")


if __name__ == "__main__":
    unittest.main()
