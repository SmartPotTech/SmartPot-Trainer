"""Native export of learned models to the `smartpot.learned-models/1` JSON contract.

The exporter writes a deterministic, tamper-evident package that SmartPot-AI can evaluate without
pickle/joblib: the three champion pipelines, the K-Means operating states, the Isolation Forest,
the LoRA adapters and exactly 16 control cases computed with scikit-learn. The package is signed
with Ed25519 and every artifact is declared with its BLAKE2b-256 digest.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

log = logging.getLogger(__name__)

SCHEMA = "smartpot.learned-models/1"
FEATURES = 10
STATE_FEATURES = 6
CONTROL_COUNT = 16
CONTROL_SEED = 20261106
MAX_ARTIFACTS = 32
PRIVATE_KEY_MODE = 0o600
PUBLIC_KEY_MODE = 0o644
PRIVATE_DIR_MODE = 0o700
TASKS = ("needs_water", "overheat", "moisture_1h")


class ExportError(RuntimeError):
    """The learned package cannot be produced without breaking the signed contract."""


def _float(value: Any, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ExportError(f"Non-finite value in {field}")
    return number


def _matrix(values: Any, field: str) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ExportError(f"Invalid matrix in {field}")
    return [[_float(item, field) for item in row] for row in array]


def public_key_hex(private_key: ed25519.Ed25519PrivateKey) -> str:
    """Raw 32-byte Ed25519 public key in hexadecimal."""
    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


def get_or_create_signing_key(key_path: Path | None = None) -> ed25519.Ed25519PrivateKey:
    """Loads the Ed25519 key or creates it once; an existing unreadable key fails closed."""
    if key_path is None:
        return ed25519.Ed25519PrivateKey.generate()

    key_path = Path(key_path)
    if key_path.exists():
        try:
            key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        except Exception as error:  # noqa: BLE001
            raise ExportError(f"Could not read the Ed25519 private key at {key_path}") from error
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ExportError(f"The key at {key_path} is not an Ed25519 private key")
        _restrict(key_path, PRIVATE_KEY_MODE)
        return key

    key_path.parent.mkdir(parents=True, exist_ok=True)
    _restrict(key_path.parent, PRIVATE_DIR_MODE)
    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    try:
        key_path.write_bytes(pem)
        _restrict(key_path, PRIVATE_KEY_MODE)
        public_path = key_path.with_name("ed25519_public.pem")
        public_path.write_bytes(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        _restrict(public_path, PUBLIC_KEY_MODE)
    except OSError as error:
        raise ExportError(f"Could not persist the Ed25519 private key at {key_path}") from error
    return key


def _restrict(path: Path, mode: int) -> None:
    """Applies POSIX permissions; on Windows chmod only toggles the read-only flag."""
    try:
        os.chmod(path, mode)
    except OSError:
        log.debug("Could not apply mode %o to %s", mode, path)


def compute_blake2b(path: Path) -> str:
    """BLAKE2b-256 digest (the same one used by hashlib.blake2b(digest_size=32))."""
    hasher = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_model_to_onnx(model: Any, output_path: Path, n_features: int = FEATURES) -> bool:
    """Exports a scikit-learn estimator to ONNX as an inspection sidecar."""
    try:
        from skl2onnx import to_onnx

        sample = np.zeros((1, n_features), dtype=np.float32)
        onnx_model = to_onnx(model, sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(onnx_model.SerializeToString())
        return True
    except Exception as error:  # noqa: BLE001
        log.warning("Could not export model %s to ONNX: %s", output_path.name, error)
        return False


def sign_manifest(
    manifest_data: dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey,
    output_path: Path,
) -> Path:
    """Signs the canonical Python JSON (without signature_hex) and writes the manifest."""
    manifest_data.pop("signature_hex", None)
    manifest_data["public_key_hex"] = public_key_hex(private_key)
    canonical = json.dumps(
        manifest_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    signature = private_key.sign(canonical)
    signed = dict(manifest_data)
    signed["signature_hex"] = signature.hex()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(signed, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def verify_manifest(
    manifest_path: Path,
    expected_public_key_hex: str | None = None,
) -> bool:
    """Verifies the Ed25519 signature and every declared BLAKE2b digest.

    When `expected_public_key_hex` is given it is mandatory: the key declared inside the manifest
    is not proof of authenticity by itself.
    """
    if not Path(manifest_path).exists():
        return False
    try:
        content = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            return False
        artifacts = content.get("artifacts")
        if not isinstance(artifacts, dict) or len(artifacts) > MAX_ARTIFACTS:
            return False
        signature_hex = content.get("signature_hex")
        public_hex = content.get("public_key_hex")
        if not isinstance(signature_hex, str) or not isinstance(public_hex, str):
            return False
        if expected_public_key_hex and not hmac.compare_digest(
            public_hex.strip().lower(), expected_public_key_hex.strip().lower()
        ):
            log.warning("Manifest %s was signed by an untrusted key", Path(manifest_path).name)
            return False

        unsigned = {key: value for key, value in content.items() if key != "signature_hex"}
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
        public_key.verify(bytes.fromhex(signature_hex), canonical)

        base = Path(manifest_path).parent
        for filename, metadata in artifacts.items():
            if not isinstance(metadata, dict) or Path(filename).name != filename:
                return False
            artifact = base / filename
            if not artifact.is_file():
                log.warning("Missing artifact declared in manifest: %s", filename)
                return False
            expected_hash = metadata.get("blake2b")
            if not isinstance(expected_hash, str):
                return False
            if not hmac.compare_digest(compute_blake2b(artifact), expected_hash.lower()):
                log.warning("Hash mismatch in %s", filename)
                return False
            declared_size = metadata.get("size_bytes")
            if declared_size is not None and int(declared_size) != artifact.stat().st_size:
                log.warning("Size mismatch in %s", filename)
                return False
        return True
    except (InvalidSignature, ValueError, OSError, TypeError, KeyError) as error:
        log.warning("Manifest verification failed for %s: %s", Path(manifest_path).name, error)
        return False


def _scaler_spec(scaler: StandardScaler) -> dict[str, list[float]]:
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    return {
        "mean": [_float(value, "scaler.mean") for value in np.asarray(scaler.mean_, dtype=np.float64)],
        "scale": [_float(value, "scaler.scale") for value in scale],
    }


def _tree_spec(tree: Any, classifier: bool, field: str) -> dict[str, list[Any]]:
    body = tree.tree_ if hasattr(tree, "tree_") else tree
    left = [int(value) for value in body.children_left]
    right = [int(value) for value in body.children_right]
    feature = [int(value) for value in body.feature]
    threshold = [float(value) for value in body.threshold]
    values: list[float] = []
    for index, child in enumerate(left):
        if child < 0:
            node = body.value[index]
            raw = node[0][1] if classifier else node[0][0]
            values.append(_float(raw, field))
        else:
            values.append(0.0)
    return {
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "value": values,
    }


def _hist_tree(predictor: Any, field: str) -> dict[str, list[Any]]:
    nodes = predictor.nodes
    left: list[int] = []
    right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    values: list[float] = []
    missing: list[bool] = []
    for node in nodes:
        if bool(node["is_leaf"]):
            left.append(-1)
            right.append(-1)
            feature.append(-2)
            threshold.append(-2.0)
        else:
            left.append(int(node["left"]))
            right.append(int(node["right"]))
            feature.append(int(node["feature_idx"]))
            threshold.append(_float(node["num_threshold"], field))
        values.append(_float(node["value"], field))
        missing.append(bool(node["missing_go_to_left"]))
    return {
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "value": values,
        "missing_left": missing,
    }


def _hist_spec(model: Any, classifier: bool) -> dict[str, Any]:
    trees = [
        _hist_tree(predictor, "hist_gradient_boosting") for iteration in model._predictors for predictor in iteration
    ]
    return {
        "kind": "hist_gradient_boosting",
        "baseline": _float(np.ravel(model._baseline_prediction)[0], "baseline"),
        "trees": trees,
        "output": "probability" if classifier else "value",
    }


def _forest_spec(model: Any, classifier: bool) -> dict[str, Any]:
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        estimators = [model]
    return {
        "kind": "forest",
        "trees": [_tree_spec(estimator, classifier, "forest") for estimator in estimators],
    }


def _knn_spec(model: Any, classifier: bool) -> dict[str, Any]:
    fit_x = np.asarray(model._fit_X, dtype=np.float64)
    targets = np.asarray(model._y, dtype=np.float64).ravel()
    k = max(1, min(int(model.n_neighbors), len(fit_x)))
    return {
        "kind": "knn",
        "k": k,
        "fit_x": _matrix(fit_x, "knn.fit_x"),
        "targets": [_float(value, "knn.targets") for value in targets],
        "output": "probability" if classifier else "value",
    }


def _dummy_spec(model: Any, classifier: bool) -> dict[str, Any]:
    if classifier:
        classes = np.asarray(model.classes_)
        priors = np.asarray(model.class_prior_)
        value = 0.0
        match = np.flatnonzero(classes == 1)
        if match.size:
            value = float(priors[int(match[0])])
        return {"kind": "constant", "value": _float(value, "constant")}
    constant = np.asarray(model.constant_).ravel()
    return {"kind": "constant", "value": _float(constant[0], "constant")}


def _model_spec(estimator: Any) -> dict[str, Any]:
    if isinstance(estimator, Pipeline):
        scaler = estimator.named_steps.get("scaler")
        model = estimator.steps[-1][1]
        spec: dict[str, Any] = {"model": _model_spec(model)}
        if isinstance(scaler, StandardScaler):
            spec["scaler"] = _scaler_spec(scaler)
        return spec
    if isinstance(estimator, LogisticRegression):
        if np.asarray(estimator.coef_).shape[0] != 1:
            raise ExportError("Only binary logistic regression is supported")
        return {
            "kind": "logistic",
            "coef": [_float(value, "logistic.coef") for value in np.asarray(estimator.coef_)[0]],
            "intercept": _float(np.asarray(estimator.intercept_)[0], "logistic.intercept"),
        }
    if isinstance(estimator, Ridge):
        return {
            "kind": "linear",
            "coef": [_float(value, "linear.coef") for value in np.asarray(estimator.coef_)],
            "intercept": _float(estimator.intercept_, "linear.intercept"),
        }
    if isinstance(estimator, MLPClassifier):
        return {
            "kind": "mlp",
            "hidden_activation": estimator.activation,
            "output_activation": estimator.out_activation_,
            "layers": [
                {
                    "weights": _matrix(weights, "mlp.weights"),
                    "bias": [_float(value, "mlp.bias") for value in np.asarray(bias)],
                }
                for weights, bias in zip(estimator.coefs_, estimator.intercepts_, strict=True)
            ],
        }
    if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
        return _forest_spec(estimator, isinstance(estimator, RandomForestClassifier))
    if isinstance(estimator, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return _forest_spec(estimator, isinstance(estimator, DecisionTreeClassifier))
    if isinstance(estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        return _hist_spec(estimator, isinstance(estimator, HistGradientBoostingClassifier))
    if isinstance(estimator, (KNeighborsClassifier, KNeighborsRegressor)):
        return _knn_spec(estimator, isinstance(estimator, KNeighborsClassifier))
    if isinstance(estimator, (DummyClassifier, DummyRegressor)):
        return _dummy_spec(estimator, isinstance(estimator, DummyClassifier))
    raise ExportError(f"Unsupported estimator: {type(estimator).__name__}")


def _states_spec(states: Any) -> dict[str, Any] | None:
    if states is None:
        return None
    return {
        "scaler": _scaler_spec(states.scaler),
        "centers": _matrix(states.kmeans.cluster_centers_, "states.centers"),
        "mapping": [[int(raw), int(target)] for raw, target in sorted(states.mapping.items())],
        "clusters": [
            {
                "id": int(cluster.id),
                "label": str(cluster.label),
                "description": str(cluster.description),
                "share": _float(cluster.share, "states.share"),
            }
            for cluster in states.clusters
        ],
    }


def _anomaly_spec(anomaly: Any) -> dict[str, Any] | None:
    if anomaly is None:
        return None
    forest: IsolationForest = anomaly.forest
    from sklearn.ensemble._iforest import _average_path_length  # noqa: PLC0415

    trees: list[dict[str, list[Any]]] = []
    for tree, depths, lengths in zip(
        forest.estimators_,
        forest._decision_path_lengths,
        forest._average_path_length_per_tree,
        strict=True,
    ):
        body = tree.tree_
        trees.append(
            {
                "left": [int(value) for value in body.children_left],
                "right": [int(value) for value in body.children_right],
                "feature": [int(value) for value in body.feature],
                "threshold": [float(value) for value in body.threshold],
                "value": [_float(value, "anomaly.value") for value in np.asarray(depths) + np.asarray(lengths) - 1.0],
            }
        )
    denominator = len(forest.estimators_) * float(_average_path_length([forest._max_samples])[0])
    return {
        "samples": int(anomaly.samples),
        "forest": {
            "trees": trees,
            "denominator": _float(denominator, "anomaly.denominator"),
            "offset": _float(forest.offset_, "anomaly.offset"),
        },
    }


def _lora_spec(adapter: Any) -> dict[str, Any]:
    d_in = int(adapter.d_in)
    k_out = int(adapter.k_out)
    rank = int(adapter.rank)
    if d_in != FEATURES or k_out != 1 or rank <= 0:
        raise ExportError("LoRA adapter dimensions do not match the learned-model contract")
    w0 = np.asarray(adapter.w0, dtype=np.float64)
    b0 = np.asarray(adapter.b0, dtype=np.float64)
    a = np.asarray(adapter.a, dtype=np.float64)
    b = np.asarray(adapter.b, dtype=np.float64)
    bias = np.asarray(adapter.bias, dtype=np.float64)
    if w0.shape != (d_in, k_out) or b0.shape != (1, k_out) or a.shape != (rank, k_out):
        raise ExportError("LoRA adapter matrices have invalid shapes")
    if b.shape != (d_in, rank) or bias.shape != (1, k_out):
        raise ExportError("LoRA adapter matrices have invalid shapes")
    return {
        "d_in": d_in,
        "k_out": k_out,
        "rank": rank,
        "alpha": _float(adapter.alpha, "lora.alpha"),
        "w0": _matrix(w0, "lora.w0"),
        "b0": _matrix(b0, "lora.b0"),
        "a": _matrix(a, "lora.a"),
        "b": _matrix(b, "lora.b"),
        "bias": _matrix(bias, "lora.bias"),
        "updates_count": int(adapter.updates_count),
    }


def export_lora_adapters(adapters: dict[str, Any], output_path: Path) -> bool:
    """Writes the exact `Adapter` shape used by Rust (w0/b0/a/b/bias/updates_count)."""
    try:
        payload = {task: _lora_spec(adapter) for task, adapter in adapters.items()}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return True
    except (ExportError, OSError, ValueError) as error:
        log.warning("Could not export LoRA adapters to %s: %s", output_path.name, error)
        return False


def control_vectors() -> np.ndarray:
    """The 16 fixed input vectors shared by the exporter and the parity tests."""
    return np.random.default_rng(CONTROL_SEED).uniform(0.0, 1.0, size=(CONTROL_COUNT, FEATURES))


def _probability(estimator: Any, row: np.ndarray) -> float:
    probabilities = np.asarray(estimator.predict_proba(row.reshape(1, -1)))[0]
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    match = np.flatnonzero(classes == 1)
    if match.size == 0:
        return 0.0
    return _float(probabilities[int(match[0])], "check.probability")


def _checks(state: Any, vectors: np.ndarray) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for vector in vectors:
        row: dict[str, Any] = {"x": [_float(value, "check.x") for value in vector]}
        for task in TASKS:
            card = state.cards.get(task) if getattr(state, "cards", None) else None
            estimator = getattr(card, "estimator", None) if card is not None else None
            if card is None or not getattr(card, "usable", False) or estimator is None:
                continue
            if task == "moisture_1h":
                row[task] = _float(np.asarray(estimator.predict(vector.reshape(1, -1)))[0], f"check.{task}")
            else:
                row[task] = _probability(estimator, vector)
        if state.states is not None:
            row["state"] = int(state.states.assign(vector[:STATE_FEATURES].reshape(1, -1)).id)
        if state.anomaly is not None:
            decision = state.anomaly.forest.decision_function(vector[:STATE_FEATURES].reshape(1, -1))
            row["anomaly"] = _float(np.asarray(decision)[0], "check.anomaly")
        checks.append(row)
    if len(checks) != CONTROL_COUNT:
        raise ExportError(f"Expected {CONTROL_COUNT} control cases, got {len(checks)}")
    return checks


def _bundle(state: Any, crop_type: str, min_samples: int | None) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    for task in TASKS:
        card = state.cards.get(task) if getattr(state, "cards", None) else None
        if card is None:
            continue
        entry: dict[str, Any] = {
            "task": task,
            "label": str(card.label),
            "status": str(card.status),
            "metric": str(card.metric),
        }
        if card.usable:
            entry["model"] = card.model
            entry["holdout"] = None if card.holdout is None else _float(card.holdout, "card.holdout")
            entry["estimator"] = _model_spec(card.estimator)
        cards.append(entry)
    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "crop_type": crop_type,
        "sklearn": sklearn.__version__,
        "created_at": state.trained_at if state.trained_at is not None else time.time(),
        "trained_at": None if state.trained_at is None else _float(state.trained_at, "trained_at"),
        "readings_at_training": int(state.readings_at_training),
        "features": FEATURES,
        "cards": cards,
        "states": _states_spec(state.states),
        "anomaly": _anomaly_spec(state.anomaly),
        "lora": {task: _lora_spec(adapter) for task, adapter in (state.lora_adapters or {}).items()},
        "checks": _checks(state, control_vectors()),
    }
    if min_samples is not None:
        bundle["min_samples"] = int(min_samples)
    return bundle


def export_crop_models(
    models_dir: Path,
    crop_type: str,
    state: Any,
    private_key: ed25519.Ed25519PrivateKey,
    min_samples: int | None = None,
    onnx: bool = False,
    trusted_public_key_hex: str | None = None,
) -> dict[str, Any]:
    """Exports and signs the native package plus optional inspection sidecars."""
    crop = crop_type.upper()
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = _bundle(state, crop, min_samples)

    inference_path = models_dir / f"{crop}_inference.json"
    inference_path.write_text(
        json.dumps(bundle, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    artifacts: dict[str, dict[str, Any]] = {
        inference_path.name: {
            "blake2b": compute_blake2b(inference_path),
            "size_bytes": inference_path.stat().st_size,
            "format": "SMARTPOT_JSON",
        }
    }

    if onnx:
        for task in TASKS:
            card = state.cards.get(task) if getattr(state, "cards", None) else None
            if card is None or not getattr(card, "usable", False):
                continue
            onnx_path = models_dir / f"{crop}_{task}.onnx"
            if export_model_to_onnx(card.estimator, onnx_path, n_features=FEATURES):
                artifacts[onnx_path.name] = {
                    "blake2b": compute_blake2b(onnx_path),
                    "size_bytes": onnx_path.stat().st_size,
                    "task": task,
                    "format": "ONNX",
                }

    if getattr(state, "lora_adapters", None):
        lora_path = models_dir / f"{crop}_lora.json"
        if export_lora_adapters(state.lora_adapters, lora_path):
            artifacts[lora_path.name] = {
                "blake2b": compute_blake2b(lora_path),
                "size_bytes": lora_path.stat().st_size,
                "format": "LORA_JSON",
            }

    joblib_path = models_dir / f"{crop}.joblib"
    if joblib_path.exists():
        artifacts[joblib_path.name] = {
            "blake2b": compute_blake2b(joblib_path),
            "size_bytes": joblib_path.stat().st_size,
            "format": "JOBLIB",
        }

    manifest = {
        "schema_version": "1.0",
        "crop_type": crop,
        "created_at": time.time(),
        "artifacts": artifacts,
    }
    manifest_path = models_dir / f"{crop}_manifest.json"
    sign_manifest(manifest, private_key, manifest_path)
    sign_manifest(manifest, private_key, models_dir / "manifest.json")
    if trusted_public_key_hex and not verify_manifest(manifest_path, trusted_public_key_hex):
        raise ExportError("The signed manifest does not verify with the trusted public key")
    return manifest
