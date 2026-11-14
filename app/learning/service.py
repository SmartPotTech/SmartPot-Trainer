"""Continuous learning and model training service: stores real readings,
runs Champion-Challenger model tournaments, K-Means, and Isolation Forest,
manages LoRA adapters, and persists joblib/ONNX models while notifying SmartPot-AI.
"""

import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import sklearn

from app.knowledge.profiles import PROFILES, get_profile
from app.learning.exporter import export_crop_models, get_or_create_signing_key, verify_manifest
from app.learning.features import build, to_store
from app.learning.lora import LowRankCropAdapter
from app.learning.quality import assess
from app.learning.store import ReadingStore, Series, StoredReading
from app.learning.trainer import AnomalyModel, ModelCard, StatesModel, train_all

log = logging.getLogger(__name__)

COUNTS_TTL = 30.0
PRUNE_EVERY = 6 * 3600.0


@dataclass
class TypeState:
    cards: dict[str, ModelCard] = field(default_factory=dict)
    states: StatesModel | None = None
    anomaly: AnomalyModel | None = None
    quality: dict[str, float] = field(default_factory=dict)
    trained_at: float | None = None
    readings_at_training: int = 0
    lora_adapters: dict[str, LowRankCropAdapter] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return any(card.usable for card in self.cards.values()) or self.states is not None


@dataclass(frozen=True)
class TrainerConfig:
    data_dir: str | None
    min_samples: int = 200
    retrain_every: int = 300
    check_seconds: int = 120
    retention_days: int = 60
    max_rows: int = 200_000
    training_rows: int = 12_000
    separate_process: bool = True
    smartpot_ai_url: str = "http://ai-smartpot:8000"
    smartpot_ai_token: str = ""
    database_url: str | None = None
    timezone: str = "America/Bogota"
    model_export_onnx: bool = True
    model_signing_key_path: str | None = None
    trusted_model_public_key_hex: str | None = None


def fit_type(crop_type: str, series: Series, min_samples: int, current: TypeState | None) -> TypeState:
    """Full training pipeline for a crop species (can run in an isolated spawned process)."""
    profile = get_profile(crop_type)
    report = assess(series.values, series.times, series.crops)
    clean = series.subset(report.keep)
    dataset = build(profile, clean)
    models = train_all(dataset, min_samples, dict(current.cards) if current else {})
    return TypeState(
        cards={task: models[task] for task in ("needs_water", "overheat", "moisture_1h")},
        states=models["states"] or (current.states if current else None),
        anomaly=models["anomaly"] or (current.anomaly if current else None),
        quality={
            "rows": report.rows,
            "completeness": report.completeness,
            "validity": report.validity,
            "outliers": report.outliers,
            "score": report.score,
        },
        trained_at=time.time(),
        readings_at_training=len(series),
        lora_adapters=dict(current.lora_adapters) if current else {},
    )


class TrainerService:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.store = ReadingStore(config.data_dir, database_url=config.database_url, timezone=config.timezone)
        self.models_dir = Path(config.data_dir) / "models" if self.store.persistent else None
        # La clave privada vive fuera de /data/models y nunca en el volumen que lee la IA.
        key_path = (
            Path(config.model_signing_key_path)
            if config.model_signing_key_path
            else (Path(config.data_dir) / "keys" / "ed25519_private.pem" if config.data_dir else None)
        )
        self._signing_key = get_or_create_signing_key(key_path)
        self.types: dict[str, TypeState] = {}
        self.training: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ProcessPoolExecutor | None = None
        self._counts: tuple[float, dict] = (0.0, {})
        self._pruned_at = 0.0
        self._load()

    # --- Ciclo de vida ---

    def start(self) -> None:
        if self.config.check_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="trainer-loop", daemon=True)
        self._thread.start()
        log.info("SmartPot-Trainer: background loop started (every %d s)", self.config.check_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.store.close()
        log.info("SmartPot-Trainer: service stopped cleanly")

    def _loop(self) -> None:
        while not self._stop.wait(self.config.check_seconds):
            try:
                if time.time() - self._pruned_at > PRUNE_EVERY:
                    removed = self.store.prune(self.config.retention_days, self.config.max_rows)
                    self._pruned_at = time.time()
                    if removed:
                        log.info("SmartPot-Trainer: %d old readings pruned", removed)
                for crop_type in self.due():
                    self.train(crop_type)
            except Exception:  # noqa: BLE001
                log.exception("SmartPot-Trainer: retraining loop failed")

    # --- Datos ---

    def ingest(self, readings: list[StoredReading]) -> int:
        stored = self.store.add(readings)
        self._counts = (0.0, {})
        return stored

    def forget(self, crop_id: str) -> int:
        removed = self.store.forget(crop_id)
        self._counts = (0.0, {})
        return removed

    def prune(self) -> int:
        """Deletes readings outside the retention window or above the row cap and refreshes counts."""
        removed = self.store.prune(self.config.retention_days, self.config.max_rows)
        self._counts = (0.0, {})
        self._pruned_at = time.time()
        return removed

    def counts(self) -> dict[str, dict]:
        cached_at, counts = self._counts
        if time.time() - cached_at > COUNTS_TTL:
            counts = self.store.counts()
            self._counts = (time.time(), counts)
        return counts

    def _retrain_threshold(self, state: TypeState | None) -> int:
        if state is None:
            return self.config.min_samples
        cards = list(state.cards.values())
        # Reintentar con menor umbral mientras falten muestras con ventana futura de una hora.
        if not cards or any(card.status != "TRAINED" and card.samples < self.config.min_samples for card in cards):
            return min(self.config.retrain_every, max(20, self.config.min_samples // 10))
        if any(card.status != "TRAINED" for card in cards):
            return min(self.config.retrain_every, max(50, self.config.min_samples // 2))
        return self.config.retrain_every

    def due(self) -> list[str]:
        due = []
        for crop_type, info in self.counts().items():
            if crop_type in self.training:
                continue
            state = self.types.get(crop_type)
            done = state.readings_at_training if state else 0
            fresh = info["readings"] - done
            if (state is None and info["readings"] >= self.config.min_samples) or (
                state is not None and fresh >= self._retrain_threshold(state)
            ):
                due.append(crop_type)
        return due

    # --- Entrenamiento ---

    def train(self, crop_type: str) -> TypeState | None:
        with self._lock:
            if crop_type in self.training:
                return self.types.get(crop_type)
            self.training.add(crop_type)
        try:
            series = self.store.load(crop_type, self.config.training_rows)
            if len(series) == 0:
                return None
            started = time.time()
            state = self._run(crop_type, series, self.types.get(crop_type))
            with self._lock:
                self._counts = (0.0, {})
            state.readings_at_training = self.counts().get(crop_type, {}).get("readings", len(series))
            with self._lock:
                self.types[crop_type] = state
            self._save(crop_type, state)
            duration = time.time() - started
            log.info(
                "SmartPot-Trainer: learning for %s completed: %d readings in %.1f s (%s)",
                crop_type,
                len(series),
                duration,
                ", ".join(f"{t}={c.status}" for t, c in state.cards.items()),
            )
            # Notificar a SmartPot-AI para hot-reload
            self.notify_ai_reload(crop_type)
            return state
        finally:
            with self._lock:
                self.training.discard(crop_type)

    def _run(self, crop_type: str, series: Series, current: TypeState | None) -> TypeState:
        if not self.config.separate_process:
            return fit_type(crop_type, series, self.config.min_samples, current)
        try:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
            return self._executor.submit(fit_type, crop_type, series, self.config.min_samples, current).result()
        except (OSError, RuntimeError) as error:
            log.warning("SmartPot-Trainer: training in same process (%s)", error)
            self._executor = None
            return fit_type(crop_type, series, self.config.min_samples, current)

    # --- Persistencia ---

    def _save(self, crop_type: str, state: TypeState) -> None:
        if self.models_dir is None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        target = self.models_dir / f"{crop_type}.joblib"
        temporary = target.with_suffix(".tmp")
        joblib.dump({"sklearn": sklearn.__version__, "state": state}, temporary)
        temporary.replace(target)

        # El paquete nativo se exporta siempre: es el contrato que consume SmartPot-AI.
        export_crop_models(
            self.models_dir,
            crop_type,
            state,
            self._signing_key,
            min_samples=self.config.min_samples,
            onnx=self.config.model_export_onnx,
            trusted_public_key_hex=self.config.trusted_model_public_key_hex,
        )

    def _load(self) -> None:
        if self.models_dir is None or not self.models_dir.exists():
            return
        trusted = self.config.trusted_model_public_key_hex
        if not trusted:
            log.info("SmartPot-Trainer: persisted joblib loading disabled without TRUSTED_MODEL_PUBLIC_KEY_HEX")
            return
        for path in self.models_dir.glob("*.joblib"):
            # Solo se reconstruyen especies conocidas: un archivo suelto nunca se deserializa.
            if path.stem not in PROFILES:
                continue
            manifest = self.models_dir / f"{path.stem}_manifest.json"
            if not verify_manifest(manifest, expected_public_key_hex=trusted):
                log.warning("SmartPot-Trainer: refusing to deserialize unverified %s", path.name)
                continue
            try:
                saved = joblib.load(path)
            except Exception as error:  # noqa: BLE001
                log.warning("Could not load %s: %s", path.name, error)
                continue
            if not isinstance(saved, dict) or "state" not in saved:
                log.warning("Ignoring malformed model state in %s", path.name)
                continue
            if saved.get("sklearn") != sklearn.__version__:
                log.info(
                    "Models for %s trained with a different scikit-learn version (%s != %s): will retrain",
                    path.stem,
                    saved.get("sklearn"),
                    sklearn.__version__,
                )
                continue
            self.types[path.stem] = saved["state"]
        if self.types:
            log.info("SmartPot-Trainer: learned models loaded: %s", ", ".join(sorted(self.types)))

    def notify_ai_reload(self, crop_type: str | None = None) -> bool:
        """Notifies the SmartPot-AI service to reload the model in memory without restarting."""
        url = f"{self.config.smartpot_ai_url.rstrip('/')}/v1/learning/reload"
        headers = {"Authorization": f"Bearer {self.config.smartpot_ai_token}"}
        payload = {"cropType": crop_type} if crop_type else {}
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.is_success:
                    log.info(
                        "SmartPot-Trainer: SmartPot-AI notified of reload for %s",
                        crop_type or "all",
                    )
                    return True
                log.warning("SmartPot-Trainer: SmartPot-AI returned status %d when notifying reload", res.status_code)
                return False
        except Exception as error:  # noqa: BLE001
            log.debug("SmartPot-Trainer: could not notify SmartPot-AI at %s (%s)", url, error)
            return False

    def adapt_online(
        self,
        crop_type: str,
        task: str,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float = 0.02,
    ) -> float:
        """Ultra-fast online update (< 1 ms) via LoRA adapter."""
        state = self.types.get(crop_type)
        if state is None:
            return 0.0
        adapter = state.lora_adapters.get(task)
        if adapter is None:
            d_in = x_batch.shape[1] if x_batch.ndim > 1 else len(x_batch)
            adapter = LowRankCropAdapter(d_in=d_in, k_out=1, rank=2, alpha=1.0)
            state.lora_adapters[task] = adapter
        loss = adapter.online_update(x_batch, y_batch, lr=lr)
        self._save(crop_type, state)
        self.notify_ai_reload(crop_type)
        return loss

    # --- Estado para API ---

    def status(self) -> dict[str, Any]:
        counts = self.counts()
        crop_types = []
        for crop_type in sorted(set(counts) | set(self.types)):
            info = counts.get(crop_type, {"readings": 0, "crops": 0})
            state = self.types.get(crop_type)
            crop_types.append(
                {
                    "crop_type": crop_type,
                    "name": PROFILES[crop_type].name if crop_type in PROFILES else crop_type,
                    "readings": int(info["readings"]),
                    "crops": int(info["crops"]),
                    "new_since_training": max(0, int(info["readings"]) - (state.readings_at_training if state else 0)),
                    "training": crop_type in self.training,
                    "trained_at": state.trained_at if state else None,
                    "quality": state.quality if state else None,
                    "models": [_card(card) for card in state.cards.values()] if state else [],
                    "states": _states(state.states) if state and state.states else None,
                    "anomaly": {"samples": state.anomaly.samples, "contamination": 0.01}
                    if state and state.anomaly
                    else None,
                }
            )
        return {
            "enabled": True,
            "persistent": self.store.persistent,
            "stored_readings": sum(int(info["readings"]) for info in counts.values()),
            "min_samples": self.config.min_samples,
            "retrain_every": self.config.retrain_every,
            "crop_types": crop_types,
        }


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _card(card: ModelCard) -> dict[str, Any]:
    return {
        "task": card.task,
        "label": card.label,
        "status": card.status,
        "reason": card.reason,
        "metric": card.metric,
        "model": card.model,
        "score": card.score,
        "std": card.std,
        "holdout": card.holdout,
        "baseline": card.baseline,
        "samples": card.samples,
        "positives": card.positives,
        "version": card.version,
        "trained_at": card.trained_at,
        "params": {k: _plain(v) for k, v in card.params.items()},
        "candidates": [
            {"model": c.model, "score": _nan_to_none(c.score), "std": _nan_to_none(c.std)} for c in card.candidates
        ],
    }


def _states(states: StatesModel) -> dict[str, Any]:
    return {
        "k": states.k,
        "silhouette": states.silhouette,
        "clusters": [{"label": c.label, "description": c.description, "share": c.share} for c in states.clusters],
    }


def _nan_to_none(value: float | None) -> float | None:
    return None if value is None or np.isnan(value) else value


def to_stored(
    crop_id: str, crop_type: str, measured_at: float, local_hour: int | None, measures: dict[str, float | None]
) -> StoredReading:
    return StoredReading(
        crop_id=crop_id,
        crop_type=crop_type,
        measured_at=measured_at,
        local_hour=local_hour,
        values=to_store(measures),
    )
