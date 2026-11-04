"""Unit tests for ONNX export, Ed25519 signing, BLAKE2b integrity, and ADBC fallback."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.learning.exporter import (
    compute_blake2b,
    export_crop_models,
    export_lora_adapters,
    export_model_to_onnx,
    get_or_create_signing_key,
    sign_manifest,
    verify_manifest,
)
from app.learning.lora import LowRankCropAdapter
from app.learning.service import TypeState
from app.learning.store import ReadingStore
from app.learning.trainer import ModelCard


def test_ed25519_key_generation_and_loading(tmp_path: Path):
    key_path = tmp_path / "keys" / "ed25519_private.pem"
    key1 = get_or_create_signing_key(key_path)
    assert key_path.exists()
    assert (tmp_path / "keys" / "ed25519_public.pem").exists()

    key2 = get_or_create_signing_key(key_path)
    # Ambas deben tener la misma clave pública
    assert key1.public_key().public_bytes_raw() == key2.public_key().public_bytes_raw()


def test_export_model_to_onnx(tmp_path: Path):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    x = np.random.randn(20, 14).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    pipe.fit(x, y)

    onnx_file = tmp_path / "test_model.onnx"
    ok = export_model_to_onnx(pipe, onnx_file, n_features=14)
    assert ok is True
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 0


def test_export_lora_adapters(tmp_path: Path):
    adapter = LowRankCropAdapter(d_in=8, k_out=1, rank=2, alpha=1.0)
    adapters = {"needs_water": adapter}
    out_file = tmp_path / "lora.json"
    ok = export_lora_adapters(adapters, out_file)
    assert ok is True
    assert out_file.exists()
    hash_val = compute_blake2b(out_file)
    assert len(hash_val) == 64


def test_manifest_signing_and_tamper_detection(tmp_path: Path):
    key = get_or_create_signing_key(tmp_path / "ed25519.pem")
    dummy_artifact = tmp_path / "artifact.bin"
    dummy_artifact.write_bytes(b"smartpot-model-bytes-v1")

    manifest_data = {
        "schema_version": "1.0",
        "crop_type": "LETTUCE",
        "artifacts": {
            "artifact.bin": {
                "blake2b": compute_blake2b(dummy_artifact),
                "size_bytes": dummy_artifact.stat().st_size,
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    sign_manifest(manifest_data, key, manifest_path)
    assert manifest_path.exists()

    # Verificación válida
    assert verify_manifest(manifest_path) is True

    # Detección de manipulación (tampering): alteramos el artefacto
    dummy_artifact.write_bytes(b"tampered-bytes")
    assert verify_manifest(manifest_path) is False

    # Restauramos artefacto pero alteramos el archivo de manifiesto
    dummy_artifact.write_bytes(b"smartpot-model-bytes-v1")
    manifest_path.write_text(manifest_path.read_text().replace("LETTUCE", "TOMATO"))
    assert verify_manifest(manifest_path) is False


def test_export_crop_models_end_to_end(tmp_path: Path):
    key = get_or_create_signing_key(tmp_path / "key.pem")
    models_dir = tmp_path / "models"

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    x = np.random.randn(20, 14).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    pipe.fit(x, y)

    card = ModelCard(
        task="needs_water",
        label="Irrigation",
        metric="f1",
        status="TRAINED",
        estimator=pipe,
    )
    adapter = LowRankCropAdapter(d_in=14, k_out=1, rank=2)
    state = TypeState(
        cards={"needs_water": card},
        lora_adapters={"needs_water": adapter},
    )

    manifest = export_crop_models(models_dir, "LETTUCE", state, key)
    assert manifest["crop_type"] == "LETTUCE"
    assert verify_manifest(models_dir / "LETTUCE_manifest.json") is True
    assert verify_manifest(models_dir / "manifest.json") is True


def test_adbc_fallback_in_reading_store(tmp_path: Path):
    # Probar que cuando database_url es inalcanzable, fallback a SQLite sin lanzar excepciones
    store = ReadingStore(tmp_path, database_url="postgresql://invalid:5432/nonexistent")
    assert store.persistent is True
    assert store.total() == 0
    counts = store.counts()
    assert isinstance(counts, dict)
    df = store.load_df("LETTUCE", 100)
    assert df.is_empty()
    store.close()
