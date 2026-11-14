"""Enterprise cryptographic compliance engine: detached signatures, hash-chained SQLite sink, encrypted queues & evidence bundles."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# RFC 3526 2048-bit MODP Group 14 safe prime (p = 2q + 1); g = 4 generates prime-order-q subgroup.
_MODP_2048_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
_MODP_2048_Q = (_MODP_2048_P - 1) // 2
_MODP_2048_G = 4


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def generate_signing_keypair(seed: bytes | None = None) -> tuple[str, str]:
    """Generates a deterministic-capable asymmetric signing keypair (Ed25519 or RFC-3526 Schnorr)."""
    raw_seed = hashlib.sha256(seed).digest() if seed is not None else os.urandom(32)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )

        priv = Ed25519PrivateKey.from_private_bytes(raw_seed)
        pub = priv.public_key()
        priv_hex = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
        pub_hex = pub.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        return f"ed25519:{priv_hex}", f"ed25519:{pub_hex}"
    except ImportError:
        x = (int.from_bytes(hashlib.sha512(raw_seed).digest(), "big") % (_MODP_2048_Q - 1)) + 1
        y = pow(_MODP_2048_G, x, _MODP_2048_P)
        return f"schnorr2048:{x:x}", f"schnorr2048:{y:x}"


def sign_detached_receipt(receipt: dict[str, Any], private_key_hex: str) -> dict[str, str]:
    """Produces a detached asymmetric signature envelope for an audit receipt."""
    msg = _canonical_json_bytes(receipt)
    if private_key_hex.startswith("ed25519:"):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        priv_bytes = bytes.fromhex(private_key_hex.split(":", 1)[1])
        priv = Ed25519PrivateKey.from_private_bytes(priv_bytes)
        sig = priv.sign(msg).hex()
        pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        return {"algorithm": "ed25519", "signature": sig, "public_key": f"ed25519:{pub}"}

    raw_x = private_key_hex.split(":", 1)[1] if ":" in private_key_hex else private_key_hex
    x = int(raw_x, 16)
    y = pow(_MODP_2048_G, x, _MODP_2048_P)
    k = (int.from_bytes(hmac.new(raw_x.encode("ascii"), msg, hashlib.sha256).digest(), "big") % (_MODP_2048_Q - 1)) + 1
    r = pow(_MODP_2048_G, k, _MODP_2048_P)
    e = int.from_bytes(hashlib.sha256(f"{r:x}:{y:x}:".encode("ascii") + msg).digest(), "big") % _MODP_2048_Q
    s = (k - e * x) % _MODP_2048_Q
    return {"algorithm": "schnorr2048", "signature": f"{r:x}:{s:x}", "public_key": f"schnorr2048:{y:x}"}


def verify_detached_receipt(receipt: dict[str, Any], detached_sig: dict[str, str]) -> bool:
    """Verifies a detached Ed25519 or Schnorr-2048 signature against a canonicalized audit receipt."""
    alg = detached_sig.get("algorithm", "")
    sig_str = detached_sig.get("signature", "")
    pub_str = detached_sig.get("public_key", "")
    if not alg or not sig_str or not pub_str:
        return False
    msg = _canonical_json_bytes(receipt)
    try:
        if alg == "ed25519":
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            pub_bytes = bytes.fromhex(pub_str.split(":", 1)[1])
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            try:
                pub.verify(bytes.fromhex(sig_str), msg)
                return True
            except InvalidSignature:
                return False

        if alg == "schnorr2048":
            y_hex = pub_str.split(":", 1)[1] if ":" in pub_str else pub_str
            r_hex, s_hex = sig_str.split(":", 1)
            y = int(y_hex, 16)
            r = int(r_hex, 16)
            s = int(s_hex, 16)
            e = int.from_bytes(hashlib.sha256(f"{r:x}:{y:x}:".encode("ascii") + msg).digest(), "big") % _MODP_2048_Q
            rv = (pow(_MODP_2048_G, s, _MODP_2048_P) * pow(y, e, _MODP_2048_P)) % _MODP_2048_P
            return hmac.compare_digest(f"{rv:x}", f"{r:x}")
    except (ValueError, KeyError, ImportError):
        return False
    return False


class AuditSinkSQLite:
    """Append-only hash-chained SQLite audit sink for tamper-evident receipt persistence."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS audit_chain ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "created_at REAL NOT NULL, "
                "prev_hash TEXT NOT NULL, "
                "row_hash TEXT NOT NULL, "
                "payload_json TEXT NOT NULL)"
            )
            conn.commit()

    def append(self, receipt: dict[str, Any]) -> str:
        payload_json = _canonical_json_bytes(receipt).decode("utf-8")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT row_hash FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = str(row[0]) if row else "GENESIS"
            row_hash = hashlib.sha256(f"{prev_hash}:{payload_json}".encode("utf-8")).hexdigest()
            conn.execute(
                "INSERT INTO audit_chain (created_at, prev_hash, row_hash, payload_json) VALUES (?, ?, ?, ?)",
                (time.time(), prev_hash, row_hash, payload_json),
            )
            conn.commit()
        return row_hash

    def list_receipts(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT seq, created_at, prev_hash, row_hash, payload_json "
                "FROM audit_chain ORDER BY seq ASC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "seq": int(seq),
                "created_at": float(ts),
                "prev_hash": str(prev_h),
                "row_hash": str(row_h),
                "receipt": json.loads(payload),
            }
            for seq, ts, prev_h, row_h, payload in rows
        ]

    def verify_chain(self) -> bool:
        expected_prev = "GENESIS"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT prev_hash, row_hash, payload_json FROM audit_chain ORDER BY seq ASC"
            ).fetchall()
        for prev_h, row_h, payload in rows:
            if not hmac.compare_digest(str(prev_h), expected_prev):
                return False
            recomputed = hashlib.sha256(f"{prev_h}:{payload}".encode("utf-8")).hexdigest()
            if not hmac.compare_digest(str(row_h), recomputed):
                return False
            expected_prev = str(row_h)
        return True


def _derive_stream_and_mac_keys(key_bytes: bytes, nonce: bytes, length: int) -> tuple[bytes, bytes]:
    dk = hashlib.pbkdf2_hmac("sha256", key_bytes, nonce, 10_000, dklen=length + 32)
    return dk[:length], dk[length:]


def encrypt_field_envelope(plaintext: str, key: str | bytes) -> dict[str, str]:
    """Encrypts and authenticates sensitive text using PBKDF2-SHA256 stream + HMAC-SHA256 tag."""
    pt_bytes = plaintext.encode("utf-8")
    key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    nonce = os.urandom(16)
    stream, mac_key = _derive_stream_and_mac_keys(key_bytes, nonce, len(pt_bytes))
    ct_bytes = bytes(b ^ s for b, s in zip(pt_bytes, stream, strict=True)) if pt_bytes else b""
    tag = hmac.new(mac_key, nonce + ct_bytes, hashlib.sha256).hexdigest()
    return {
        "alg": "PBKDF2-SHA256-HMAC",
        "nonce": nonce.hex(),
        "ciphertext": ct_bytes.hex(),
        "tag": tag,
    }


def decrypt_field_envelope(envelope: dict[str, str], key: str | bytes) -> str:
    """Verifies envelope integrity before decrypting field ciphertext."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    nonce = bytes.fromhex(envelope["nonce"])
    ct_bytes = bytes.fromhex(envelope["ciphertext"])
    stream, mac_key = _derive_stream_and_mac_keys(key_bytes, nonce, len(ct_bytes))
    expected_tag = hmac.new(mac_key, nonce + ct_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_tag, envelope.get("tag", "")):
        raise ValueError("Envelope authentication failed")
    pt_bytes = bytes(b ^ s for b, s in zip(ct_bytes, stream, strict=True)) if ct_bytes else b""
    return pt_bytes.decode("utf-8")


def _compute_merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"EMPTY_MERKLE_TREE").hexdigest()
    nodes = [hashlib.sha256(leaf.encode("utf-8")).hexdigest() for leaf in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(f"{nodes[i]}:{nodes[i + 1]}".encode("utf-8")).hexdigest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def export_evidence_bundle(
    receipts: list[dict[str, Any]],
    out_path: Path | str,
    *,
    hmac_secret: str | None = None,
    frameworks: list[str] | None = None,
) -> dict[str, Any]:
    """Exports a Merkle-rooted SOC2/HIPAA/GDPR compliance evidence bundle to JSON."""
    leaves = [_canonical_json_bytes(r).decode("utf-8") for r in receipts]
    merkle_root = _compute_merkle_root(leaves)
    bundle: dict[str, Any] = {
        "schema_version": "ncm-evidence-v1",
        "frameworks": frameworks or ["SOC2", "HIPAA", "GDPR"],
        "generated_at": round(time.time(), 3),
        "total_receipts": len(receipts),
        "merkle_root_sha256": merkle_root,
        "receipts": receipts,
    }
    if hmac_secret:
        bundle["bundle_hmac_sha256"] = hmac.new(
            hmac_secret.encode("utf-8"),
            merkle_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def verify_evidence_bundle(
    bundle_path: Path | str,
    *,
    hmac_secret: str | None = None,
) -> dict[str, Any]:
    """Verifies Merkle root and optional HMAC seal of an exported compliance evidence bundle."""
    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    receipts = payload.get("receipts", [])
    leaves = [_canonical_json_bytes(r).decode("utf-8") for r in receipts]
    recomputed_root = _compute_merkle_root(leaves)
    merkle_ok = hmac.compare_digest(recomputed_root, str(payload.get("merkle_root_sha256", "")))
    hmac_ok = True
    if hmac_secret is not None:
        sig = str(payload.get("bundle_hmac_sha256", ""))
        expected = hmac.new(
            hmac_secret.encode("utf-8"),
            recomputed_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        hmac_ok = bool(sig) and hmac.compare_digest(sig, expected)
    return {
        "valid": bool(merkle_ok and hmac_ok),
        "merkle_ok": merkle_ok,
        "hmac_ok": hmac_ok,
        "total_receipts": len(receipts),
        "merkle_root_sha256": recomputed_root,
        "frameworks": payload.get("frameworks", []),
    }
