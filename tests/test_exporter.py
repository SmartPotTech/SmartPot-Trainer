"""Native learned-model export, Ed25519 signing, BLAKE2b integrity and key custody."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.learning.exporter import (
    SCHEMA,
    ExportError,
    _model_spec,
    compute_blake2b,
    control_vectors,
    export_crop_models,
    export_lora_adapters,
    export_model_to_onnx,
    get_or_create_signing_key,
    public_key_hex,
    sign_manifest,
    verify_manifest,
)
from app.learning.lora import LowRankCropAdapter
from app.learning.service import TypeState
from app.learning.trainer import ModelCard, train_anomaly, train_states


def pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def learned_state() -> TypeState:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(140, 10))
    y = (x[:, 0] + rng.normal(scale=0.3, size=140) > 0).astype(int)
    logistic = pipeline(LogisticRegression(max_iter=1000)).fit(x, y)
    forest = pipeline(RandomForestClassifier(n_estimators=20, max_depth=4, random_state=7)).fit(x, y)
    ridge = pipeline(Ridge()).fit(x, rng.normal(size=140))
    positions = rng.uniform(0.0, 1.0, size=(140, 6))
    cards = {
        "needs_water": ModelCard(
            task="needs_water",
            label="Will the substrate dry out in the next hour?",
            status="TRAINED",
            metric="Macro F1",
            model="Logistic regression",
            holdout=0.91,
            estimator=logistic,
        ),
        "overheat": ModelCard(
            task="overheat",
            label="Will the temperature exceed the maximum in the next hour?",
            status="TRAINED",
            metric="Macro F1",
            model="Random forest",
            holdout=0.84,
            estimator=forest,
        ),
        "moisture_1h": ModelCard(
            task="moisture_1h",
            label="Substrate moisture in one hour",
            status="TRAINED",
            metric="Mean absolute error (%)",
            model="Ridge regression",
            holdout=2.5,
            estimator=ridge,
        ),
    }
    return TypeState(
        cards=cards,
        states=train_states(positions, np.full(140, 12.0)),
        anomaly=train_anomaly(positions),
        trained_at=1_790_000_000.0,
        readings_at_training=140,
        lora_adapters={"needs_water": LowRankCropAdapter(d_in=10, rank=2, alpha=1.0)},
    )


def test_ed25519_key_generation_and_loading(tmp_path: Path):
    key_path = tmp_path / "keys" / "ed25519_private.pem"
    key1 = get_or_create_signing_key(key_path)
    assert key_path.exists()
    assert (tmp_path / "keys" / "ed25519_public.pem").exists()

    key2 = get_or_create_signing_key(key_path)
    assert key1.public_key().public_bytes_raw() == key2.public_key().public_bytes_raw()
    if os.name == "posix":
        assert (key_path.stat().st_mode & 0o777) == 0o600


def test_corrupt_existing_key_fails_closed(tmp_path: Path):
    key_path = tmp_path / "ed25519_private.pem"
    key_path.write_text("not a pem key", encoding="utf-8")
    with pytest.raises(ExportError):
        get_or_create_signing_key(key_path)


def test_export_model_to_onnx(tmp_path: Path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=20)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(x, y)

    onnx_file = tmp_path / "test_model.onnx"
    assert export_model_to_onnx(model, onnx_file, n_features=10) is True
    assert onnx_file.stat().st_size > 0


def test_export_lora_adapters_uses_the_rust_contract(tmp_path: Path):
    adapter = LowRankCropAdapter(d_in=10, k_out=1, rank=2, alpha=1.0)
    adapter.online_update(np.ones((2, 10)), np.array([1.0, 0.0]))
    out_file = tmp_path / "lora.json"
    assert export_lora_adapters({"needs_water": adapter}, out_file) is True
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert set(payload["needs_water"]) == {
        "d_in",
        "k_out",
        "rank",
        "alpha",
        "w0",
        "b0",
        "a",
        "b",
        "bias",
        "updates_count",
    }
    assert payload["needs_water"]["updates_count"] == 1


def test_manifest_signing_tamper_detection_and_pinned_key(tmp_path: Path):
    key = get_or_create_signing_key(tmp_path / "key.pem")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"smartpot-model-bytes-v1")
    manifest_data = {
        "schema_version": "1.0",
        "crop_type": "LETTUCE",
        "artifacts": {
            "artifact.bin": {
                "blake2b": compute_blake2b(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    sign_manifest(manifest_data, key, manifest_path)
    assert verify_manifest(manifest_path) is True
    assert verify_manifest(manifest_path, public_key_hex(key)) is True

    artifact.write_bytes(b"tampered-bytes")
    assert verify_manifest(manifest_path, public_key_hex(key)) is False

    artifact.write_bytes(b"smartpot-model-bytes-v1")
    manifest_path.write_text(manifest_path.read_text().replace("LETTUCE", "TOMATO"))
    assert verify_manifest(manifest_path, public_key_hex(key)) is False

    other = get_or_create_signing_key(tmp_path / "other.pem")
    assert verify_manifest(manifest_path, public_key_hex(other)) is False


def test_native_export_matches_the_learned_models_contract(tmp_path: Path):
    key = get_or_create_signing_key(tmp_path / "keys" / "ed25519_private.pem")
    state = learned_state()
    manifest = export_crop_models(
        tmp_path,
        "LETTUCE",
        state,
        key,
        min_samples=100,
        onnx=False,
        trusted_public_key_hex=public_key_hex(key),
    )

    bundle = json.loads((tmp_path / "LETTUCE_inference.json").read_text(encoding="utf-8"))
    assert bundle["schema"] == SCHEMA
    assert bundle["features"] == 10
    assert bundle["readings_at_training"] == 140
    assert bundle["min_samples"] == 100
    assert len(bundle["checks"]) == 16
    assert len(bundle["cards"]) == 3
    assert all(len(check["x"]) == 10 for check in bundle["checks"])
    assert all(
        key in check
        for check in bundle["checks"]
        for key in ("needs_water", "overheat", "moisture_1h", "state", "anomaly")
    )
    assert bundle["cards"][0]["estimator"]["model"]["kind"] == "logistic"
    assert bundle["cards"][1]["estimator"]["model"]["kind"] == "forest"
    assert len(bundle["states"]["centers"]) == state.states.k
    assert bundle["states"]["centers"] == state.states.kmeans.cluster_centers_.tolist()
    assert len(bundle["anomaly"]["forest"]["trees"]) == 150
    assert bundle["lora"]["needs_water"]["d_in"] == 10

    assert "LETTUCE_inference.json" in manifest["artifacts"]
    assert "LETTUCE_lora.json" in manifest["artifacts"]
    assert verify_manifest(tmp_path / "LETTUCE_manifest.json", public_key_hex(key)) is True
    assert verify_manifest(tmp_path / "manifest.json", public_key_hex(key)) is True


def test_control_vectors_are_deterministic():
    first = control_vectors()
    second = control_vectors()
    assert first.shape == (16, 10)
    np.testing.assert_array_equal(first, second)


def test_every_supported_estimator_has_a_native_spec():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 10))
    y = (x[:, 0] > 0).astype(int)
    models = {
        "logistic": pipeline(LogisticRegression(max_iter=1000)).fit(x, y),
        "linear": pipeline(Ridge()).fit(x, y.astype(float)),
        "forest": pipeline(RandomForestClassifier(n_estimators=5, max_depth=3, random_state=7)).fit(x, y),
        "hist_gradient_boosting": pipeline(HistGradientBoostingClassifier(max_iter=3, random_state=7)).fit(x, y),
        "knn": pipeline(KNeighborsClassifier(n_neighbors=3)).fit(x, y),
        "mlp": pipeline(MLPClassifier(hidden_layer_sizes=(4,), max_iter=50, random_state=7)).fit(x, y),
        "constant": pipeline(DummyClassifier(strategy="most_frequent")).fit(x, y),
    }
    assert _model_spec(models["logistic"])["model"]["kind"] == "logistic"
    assert _model_spec(models["linear"])["model"]["kind"] == "linear"
    assert _model_spec(models["forest"])["model"]["kind"] == "forest"
    assert _model_spec(models["hist_gradient_boosting"])["model"]["kind"] == "hist_gradient_boosting"
    assert _model_spec(models["knn"])["model"]["kind"] == "knn"
    assert _model_spec(models["mlp"])["model"]["kind"] == "mlp"
    assert _model_spec(models["constant"])["model"]["kind"] == "constant"
    assert _model_spec(pipeline(DummyRegressor()).fit(x, y.astype(float)))["model"]["kind"] == "constant"
