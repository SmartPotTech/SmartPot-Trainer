"""Exportador de modelos aprendidos a formato ONNX firmado con Ed25519 y hash BLAKE2b.

Genera un manifiesto criptográfico a prueba de manipulación (anti-tampering/data poisoning)
que permite a SmartPot-AI verificar la autenticidad e integridad de los artefactos antes
de cargarlos en memoria o ejecutar inferencia.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

log = logging.getLogger(__name__)

# Muestra sintética estándar de 1 fila para inferencia de forma en ONNX
# [temp, humidity, brightness, ph, tds, atmosphere, soil_moisture, vpd, rolling_temp, ...]
DUMMY_INPUT_SAMPLE = np.zeros((1, 14), dtype=np.float32)


def get_or_create_signing_key(key_path: Path | None = None) -> ed25519.Ed25519PrivateKey:
    """Carga o genera una clave privada Ed25519 para la firma de modelos."""
    if key_path and key_path.exists():
        try:
            return serialization.load_pem_private_key(key_path.read_bytes(), password=None)  # type: ignore[return-value]
        except Exception as error:  # noqa: BLE001
            log.warning("No se pudo cargar la clave Ed25519 desde %s (%s); generando una efímera", key_path, error)

    private_key = ed25519.Ed25519PrivateKey.generate()
    if key_path:
        try:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            key_path.write_bytes(pem)
            pub_path = key_path.with_name("ed25519_public.pem")
            pub_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            pub_path.write_bytes(pub_pem)
        except OSError as error:
            log.warning("No se pudo persistir la clave privada Ed25519 en %s: %s", key_path, error)

    return private_key


def compute_blake2b(path: Path) -> str:
    """Calcula el hash criptográfico BLAKE2b (digest de 256 bits) del archivo."""
    hasher = hashlib.blake2b(digest_size=32)
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_model_to_onnx(model: Any, output_path: Path, n_features: int = 14) -> bool:
    """Exporta un modelo o pipeline de scikit-learn a formato ONNX."""
    try:
        from skl2onnx import to_onnx

        sample = np.zeros((1, n_features), dtype=np.float32)
        onx = to_onnx(model, sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(onx.SerializeToString())
        return True
    except Exception as error:  # noqa: BLE001
        log.warning("No se pudo exportar modelo %s a ONNX: %s", output_path.name, error)
        return False


def export_lora_adapters(adapters: dict[str, Any], output_path: Path) -> bool:
    """Exporta adaptadores LoRA a formato JSON estructurado."""
    try:
        payload = {}
        for task, adapter in adapters.items():
            rank = int(getattr(adapter, "rank", 2))
            alpha = float(getattr(adapter, "alpha", 1.0))
            scale = alpha / float(rank) if rank > 0 else 1.0
            a_mat = adapter.a
            b_mat = adapter.b
            delta_w = (scale * (b_mat @ a_mat)).tolist()
            payload[task] = {
                "d_in": int(adapter.d_in),
                "k_out": int(adapter.k_out),
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
                "a_matrix": a_mat.tolist(),
                "b_matrix": b_mat.tolist(),
                "delta_w": delta_w,
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception as error:  # noqa: BLE001
        log.warning("No se pudieron exportar adaptadores LoRA a %s: %s", output_path.name, error)
        return False


def sign_manifest(
    manifest_data: dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey,
    output_path: Path,
) -> Path:
    """Genera la firma Ed25519 sobre el JSON canónico del manifiesto y lo guarda en disco."""
    pub_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    manifest_data["public_key_hex"] = pub_bytes.hex()

    # Serialización determinista canónica sin espacios
    canonical_bytes = json.dumps(manifest_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = private_key.sign(canonical_bytes)

    signed_manifest = dict(manifest_data)
    signed_manifest["signature_hex"] = signature.hex()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(signed_manifest, indent=2), encoding="utf-8")
    return output_path


def verify_manifest(manifest_path: Path) -> bool:
    """Valida la firma Ed25519 y la integridad BLAKE2b de los artefactos del manifiesto."""
    if not manifest_path.exists():
        return False
    try:
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        sig_hex = content.get("signature_hex")
        pub_hex = content.get("public_key_hex")
        artifacts = content.get("artifacts", {})
        if not sig_hex or not pub_hex:
            return False

        # Reconstruir datos canónicos
        data_to_verify = {k: v for k, v in content.items() if k != "signature_hex"}
        canonical_bytes = json.dumps(data_to_verify, sort_keys=True, separators=(",", ":")).encode("utf-8")

        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pub_key.verify(bytes.fromhex(sig_hex), canonical_bytes)

        # Verificar integridad de cada artefacto
        base_dir = manifest_path.parent
        for filename, meta in artifacts.items():
            artifact_file = base_dir / filename
            if not artifact_file.exists():
                log.warning("Artefacto faltante declarado en manifiesto: %s", filename)
                return False
            expected_hash = meta.get("blake2b")
            actual_hash = compute_blake2b(artifact_file)
            if actual_hash != expected_hash:
                log.warning("Hash mismatch en %s: esperado %s != actual %s", filename, expected_hash, actual_hash)
                return False

        return True
    except (InvalidSignature, ValueError, OSError) as error:
        log.warning("Fallo en verificación de manifiesto %s: %s", manifest_path.name, error)
        return False


def export_crop_models(
    models_dir: Path,
    crop_type: str,
    state: Any,
    private_key: ed25519.Ed25519PrivateKey,
) -> dict[str, Any]:
    """Exporta y firma todos los modelos, adaptadores LoRA y manifiesto de una especie."""
    models_dir.mkdir(parents=True, exist_ok=True)
    exported_artifacts: dict[str, dict[str, Any]] = {}

    # 1. Exportar clasificadores/regresores campeones a ONNX
    if hasattr(state, "cards") and isinstance(state.cards, dict):
        for task, card in state.cards.items():
            estimator = getattr(card, "estimator", getattr(card, "pipeline", None))
            if getattr(card, "usable", False) and estimator is not None:
                onnx_name = f"{crop_type}_{task}.onnx"
                onnx_path = models_dir / onnx_name
                # Determinar dimensión de entrada
                n_feats = 14
                if hasattr(estimator, "n_features_in_"):
                    n_feats = estimator.n_features_in_
                if export_model_to_onnx(estimator, onnx_path, n_features=n_feats):
                    exported_artifacts[onnx_name] = {
                        "blake2b": compute_blake2b(onnx_path),
                        "size_bytes": onnx_path.stat().st_size,
                        "task": task,
                        "format": "ONNX",
                    }

    # 2. Exportar adaptadores LoRA
    if hasattr(state, "lora_adapters") and state.lora_adapters:
        lora_name = f"{crop_type}_lora.json"
        lora_path = models_dir / lora_name
        if export_lora_adapters(state.lora_adapters, lora_path):
            exported_artifacts[lora_name] = {
                "blake2b": compute_blake2b(lora_path),
                "size_bytes": lora_path.stat().st_size,
                "format": "LORA_JSON",
            }

    # 3. Registrar también el archivo joblib existente si está en el directorio
    joblib_name = f"{crop_type}.joblib"
    joblib_path = models_dir / joblib_name
    if joblib_path.exists():
        exported_artifacts[joblib_name] = {
            "blake2b": compute_blake2b(joblib_path),
            "size_bytes": joblib_path.stat().st_size,
            "format": "JOBLIB",
        }

    # 4. Crear manifiesto y firmar
    manifest_data = {
        "schema_version": "1.0",
        "crop_type": crop_type,
        "created_at": time.time(),
        "artifacts": exported_artifacts,
    }
    manifest_path = models_dir / f"{crop_type}_manifest.json"
    sign_manifest(manifest_data, private_key, manifest_path)

    # Actualizar manifiesto global en models_dir/manifest.json
    global_manifest_path = models_dir / "manifest.json"
    sign_manifest(manifest_data, private_key, global_manifest_path)

    return manifest_data
