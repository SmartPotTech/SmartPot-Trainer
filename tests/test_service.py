import json
import tempfile
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from app.learning.exporter import verify_manifest
from app.learning.service import TrainerConfig, TrainerService
from app.learning.store import StoredReading
from tests.learning_data import lettuce_series


def public_key_from_pem(path: Path) -> str:
    key = serialization.load_pem_public_key(path.read_bytes())
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


class AIRecorder:
    """Loopback SmartPot-AI stub that records the reload webhook and answers with a fixed status."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[dict[str, object]] = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                recorder.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(raw or b"{}"),
                    }
                )
                self.send_response(recorder.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "AIRecorder":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def steady_values(temperature: float = 20.0, soil_moisture: float = 70.0) -> dict[str, float]:
    return {
        "temperature": temperature,
        "humidity": 60.0,
        "brightness": 800.0,
        "ph": 6.0,
        "tds": 700.0,
        "atmosphere": 1012.0,
        "soil_moisture": soil_moisture,
    }


def config_for(data_dir: str, ai_url: str, **overrides: object) -> TrainerConfig:
    values: dict[str, object] = {
        "data_dir": data_dir,
        "min_samples": 100,
        "retrain_every": 50,
        "check_seconds": 0,
        "separate_process": False,  # en pruebas unitarias usamos el mismo proceso
        "smartpot_ai_url": ai_url,
        "smartpot_ai_token": "trainer-to-ai-token-for-tests",
        "model_export_onnx": False,
    }
    values.update(overrides)
    return TrainerConfig(**values)


def test_service_lifecycle_and_training():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        service = TrainerService(config_for(tmpdir, ai.url))
        service.start()

        # Ingerir lecturas
        # Muestreo de 30 min: conserva los ciclos de riego y mantiene la matriz por debajo del umbral de
        # búsqueda de hiperparámetros, para que la prueba del ciclo complete en segundos.
        readings = lettuce_series("crop-srv", days=3.0, step_minutes=30.0)
        stored = service.ingest(readings)
        assert stored == len(readings)

        # Verificar due
        due_crops = service.due()
        assert "LETTUCE" in due_crops

        # Entrenar
        state = service.train("LETTUCE")
        assert state is not None
        assert state.quality["rows"] > 0
        inference_path = Path(tmpdir) / "models" / "LETTUCE_inference.json"
        assert inference_path.exists()
        assert (Path(tmpdir) / "keys" / "ed25519_private.pem").exists()

        # El webhook de recarga viaja al servicio de IA con el token y la especie entrenada.
        assert len(ai.requests) == 1
        assert ai.requests[0]["path"] == "/v1/learning/reload"
        assert ai.requests[0]["authorization"] == "Bearer trainer-to-ai-token-for-tests"
        assert ai.requests[0]["body"] == {"cropType": "LETTUCE"}

        # El paquete nativo está firmado y el manifiesto verifica con la clave pública generada.
        bundle = json.loads(inference_path.read_text(encoding="utf-8"))
        manifest = json.loads((Path(tmpdir) / "models" / "LETTUCE_manifest.json").read_text(encoding="utf-8"))
        trusted = public_key_from_pem(Path(tmpdir) / "keys" / "ed25519_public.pem")
        assert bundle["schema"] == "smartpot.learned-models/1"
        assert bundle["crop_type"] == "LETTUCE"
        assert bundle["features"] == 10
        assert len(bundle["checks"]) == 16
        assert manifest["public_key_hex"] == trusted
        assert isinstance(manifest["signature_hex"], str) and len(manifest["signature_hex"]) == 128
        assert manifest["artifacts"]["LETTUCE_inference.json"]["blake2b"]
        assert verify_manifest(Path(tmpdir) / "models" / "LETTUCE_manifest.json", trusted) is True

        # Verificar status
        status = service.status()
        assert status["enabled"]
        assert status["persistent"]
        assert status["stored_readings"] == len(readings)
        lettuce_info = next(c for c in status["crop_types"] if c["crop_type"] == "LETTUCE")
        assert lettuce_info["readings"] == len(readings)
        assert lettuce_info["trained_at"] is not None

        # Olvido por cultivo: se borran sus lecturas y el estado entrenado sigue disponible.
        assert service.forget("crop-srv") == len(readings)
        assert service.status()["stored_readings"] == 0

        # Poda sin filas que retirar no cambia el estado.
        assert service.prune() == 0

        # Verificar persistencia en disco
        service.stop()

        # Solo se recarga un joblib autenticado con la clave pública fijada.
        verified = config_for(tmpdir, ai.url, trusted_model_public_key_hex=trusted)
        service2 = TrainerService(verified)
        assert "LETTUCE" in service2.types
        assert service2.types["LETTUCE"].quality["rows"] > 0
        service2.stop()


def test_mono_class_training_stays_pending_and_exports_a_signed_bundle():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder(status=500) as ai:
        service = TrainerService(config_for(tmpdir, ai.url))
        readings = [replace(reading, values=steady_values()) for reading in lettuce_series("crop-mono", days=0.35)]
        assert service.ingest(readings) == len(readings)

        # Sin eventos de sequía ni de calor, los clasificadores no deben romper el entrenamiento.
        state = service.train("LETTUCE")
        assert state is not None
        assert state.readings_at_training == len(readings)
        for task in ("needs_water", "overheat"):
            card = state.cards[task]
            assert card.status == "PENDING"
            assert "one class" in (card.reason or "").lower()
            assert card.estimator is None

        # El webhook falló (500), pero el paquete exportado sigue siendo válido y verificable.
        assert len(ai.requests) == 1
        assert ai.requests[0]["path"] == "/v1/learning/reload"
        inference_path = Path(tmpdir) / "models" / "LETTUCE_inference.json"
        bundle = json.loads(inference_path.read_text(encoding="utf-8"))
        assert len(bundle["checks"]) == 16
        guarded = [card for card in bundle["cards"] if card["task"] in ("needs_water", "overheat")]
        assert all(card["status"] == "PENDING" for card in guarded)
        trusted = public_key_from_pem(Path(tmpdir) / "keys" / "ed25519_public.pem")
        assert verify_manifest(Path(tmpdir) / "models" / "LETTUCE_manifest.json", trusted) is True

        # El estado en memoria no quedó corrupto: la siguiente consulta sigue respondiendo.
        status = service.status()
        lettuce_info = next(c for c in status["crop_types"] if c["crop_type"] == "LETTUCE")
        assert lettuce_info["models"]
        service.stop()


def test_training_filters_temporal_jumps_and_duplicates():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        service = TrainerService(config_for(tmpdir, ai.url, min_samples=60))
        readings = [
            replace(reading, values=steady_values())
            for reading in lettuce_series("crop-quality", days=2.0, step_minutes=30.0)
        ]
        assert service.ingest(readings) == len(readings)
        # Un reintento de la misma ingesta no duplica filas.
        assert service.ingest(readings[:50]) == 0
        assert service.store.total() == len(readings)

        # Salto térmico imposible en 30 s: dentro del rango físico, fuera de la derivada temporal.
        previous = readings[20]
        jump = replace(
            previous,
            measured_at=previous.measured_at + 30.0,
            values={**previous.values, "temperature": previous.values["temperature"] + 20.0},
        )
        assert service.ingest([jump]) == 1

        # Un valor fuera de la escala física también se excluye del conjunto de entrenamiento.
        impossible = replace(
            previous,
            measured_at=previous.measured_at + 60.0,
            values={**previous.values, "temperature": 999.0},
        )
        assert service.ingest([impossible]) == 1

        state = service.train("LETTUCE")
        assert state is not None
        assert state.quality["rows"] == len(readings) + 2
        assert state.quality["validity"] < 1.0
        assert state.quality["outliers"] >= 1
        assert state.quality["score"] < 1.0
        assert service.store.total() == len(readings) + 2
        service.stop()


def test_prune_refreshes_cached_counts_immediately():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        service = TrainerService(config_for(tmpdir, ai.url, retention_days=1))
        now = time.time()
        values = steady_values()
        old = StoredReading("crop-old", "LETTUCE", now - 10 * 86_400, 12, values)
        fresh = StoredReading("crop-fresh", "LETTUCE", now, 12, values)
        assert service.ingest([old, fresh]) == 2
        assert service.counts()["LETTUCE"]["readings"] == 2

        assert service.prune() == 1
        # counts() tenía un TTL de 30 s; la poda debe invalidarlo para no informar filas borradas.
        assert service.counts()["LETTUCE"]["readings"] == 1
        assert service.status()["stored_readings"] == 1
        service.stop()


def test_status_derives_stored_readings_from_counts_and_clamps_fresh():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        from app.learning.service import TypeState
        from app.learning.trainer import ModelCard

        service = TrainerService(config_for(tmpdir, ai.url, min_samples=100, retrain_every=50))
        try:
            service.store.counts = lambda: {
                "LETTUCE": {"readings": 203, "crops": 2},
                "BASIL": {"readings": 17, "crops": 1},
            }
            service.types["LETTUCE"] = TypeState(
                trained_at=time.time(),
                readings_at_training=250,
                cards={
                    "moisture_1h": ModelCard(
                        task="moisture_1h",
                        label="Substrate",
                        metric="mae",
                        status="PENDING",
                        reason="wait",
                        samples=0,
                    )
                },
            )
            service._counts = (0.0, {})

            status = service.status()
            assert status["stored_readings"] == 220
            assert sum(item["readings"] for item in status["crop_types"]) == 220
            lettuce = next(item for item in status["crop_types"] if item["crop_type"] == "LETTUCE")
            assert lettuce["new_since_training"] == 0
        finally:
            service.stop()


def test_due_retries_pending_models_before_retrain_every():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        from app.learning.service import TypeState
        from app.learning.trainer import ModelCard

        service = TrainerService(config_for(tmpdir, ai.url, min_samples=200, retrain_every=300))
        try:
            live_readings = 203
            service.store.counts = lambda: {"LETTUCE": {"readings": live_readings, "crops": 2}}

            def refresh_counts(n: int) -> None:
                nonlocal live_readings
                live_readings = n
                service._counts = (0.0, {})

            def pending_card(task: str, samples: int) -> ModelCard:
                return ModelCard(
                    task=task,
                    label=task,
                    metric="f1_macro",
                    status="PENDING",
                    reason=f"Insufficient readings: {samples} of 200.",
                    samples=samples,
                )

            refresh_counts(203)
            assert "LETTUCE" in service.due()
            service.types["LETTUCE"] = TypeState(
                trained_at=time.time(),
                readings_at_training=203,
                quality={"rows": 203, "completeness": 0.8571, "validity": 1.0, "outliers": 0, "score": 0.8571},
                cards={
                    "needs_water": pending_card("needs_water", 100),
                    "overheat": pending_card("overheat", 100),
                    "moisture_1h": pending_card("moisture_1h", 100),
                },
            )

            # Con 0 o 7 lecturas nuevas no debe reintentar ni girar en bucle.
            assert "LETTUCE" not in service.due()
            refresh_counts(210)
            assert "LETTUCE" not in service.due()

            # Al alcanzar 20 lecturas nuevas con modelos aún por debajo de min_samples, debe reintentar.
            refresh_counts(223)
            assert "LETTUCE" in service.due()

            # Si las tres tarjetas ya superaron el entrenamiento inicial, respeta retrain_every=300.
            service.types["LETTUCE"].cards = {
                task: ModelCard(task=task, label=task, metric="f1_macro", status="TRAINED", reason="ok", samples=220)
                for task in ("needs_water", "overheat", "moisture_1h")
            }
            assert "LETTUCE" not in service.due()
            refresh_counts(502)
            assert "LETTUCE" not in service.due()
            refresh_counts(503)
            assert "LETTUCE" in service.due()
        finally:
            service.stop()


def test_training_advances_after_horizon_window_with_missing_atmosphere():
    with tempfile.TemporaryDirectory() as tmpdir, AIRecorder() as ai:
        service = TrainerService(config_for(tmpdir, ai.url, min_samples=150, retrain_every=300))
        try:
            full = lettuce_series("crop-real-1", days=0.16, step_minutes=1.0, seed=7)
            for item in full:
                item.values["atmosphere"] = None

            initial = full[:160]
            assert service.ingest(initial) == 160
            assert "LETTUCE" in service.due()

            first = service.train("LETTUCE")
            assert first is not None
            assert first.quality["completeness"] == 0.8571
            assert first.cards["moisture_1h"].status == "PENDING"
            assert first.cards["moisture_1h"].samples > 0

            # Al completar >200 lecturas con ventana futura suficiente, reentrena antes de esperar 300 más.
            added = service.ingest(full[160:])
            assert added > 0
            assert "LETTUCE" in service.due()

            second = service.train("LETTUCE")
            assert second is not None
            assert second.cards["moisture_1h"].status == "TRAINED"
            status = service.status()
            assert status["stored_readings"] == status["crop_types"][0]["readings"]
            assert status["crop_types"][0]["new_since_training"] == 0
        finally:
            service.stop()
