import json
import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.learning.exporter import verify_manifest
from app.main import create_app
from tests.learning_data import lettuce_series
from tests.test_service import public_key_from_pem


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    """App with its own data directory so training and pruning do not touch the shared session."""

    def build(**overrides: str) -> TestClient:
        values = {
            "DATA_DIR": str(tmp_path),
            "LEARNING_MIN_SAMPLES": "80",
            "LEARNING_RETRAIN_EVERY": "50",
            "LEARNING_CHECK_SECONDS": "0",
            "LEARNING_SEPARATE_PROCESS": "false",
            "LEARNING_RETENTION_DAYS": "60",
            "MODEL_EXPORT_ONNX": "false",
            # Sin red externa: el webhook de recarga no debe frenar la prueba.
            "SMARTPOT_AI_URL": "http://127.0.0.1:9",
        }
        values.update(overrides)
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return TestClient(create_app())

    try:
        with build() as client:
            yield client
    finally:
        get_settings.cache_clear()


def payload_for(readings) -> dict:
    def measured_at(value: float) -> str:
        return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")

    return {
        "readings": [
            {
                "cropId": reading.crop_id,
                "cropType": reading.crop_type,
                "measuredAt": measured_at(reading.measured_at),
                "localHour": reading.local_hour,
                "measures": {
                    "temperature": reading.values["temperature"],
                    "humidity": reading.values["humidity"],
                    "brightness": reading.values["brightness"],
                    "ph": reading.values["ph"],
                    "tds": reading.values["tds"],
                    "atmosphere": reading.values["atmosphere"],
                    "soilMoisture": reading.values["soil_moisture"],
                },
            }
            for reading in readings
        ]
    }


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert data["service"] == "smartpot-trainer"


def test_auth_protection(client):
    res = client.get("/v1/learning/status")
    assert res.status_code == 401

    res = client.get("/v1/learning/status", headers={"Authorization": "Bearer bad-token"})
    assert res.status_code == 401

    # Una cabecera no ASCII debe fallar cerrada con 401, nunca con 500.
    res = client.get(
        "/v1/learning/status",
        headers=[(b"authorization", "Bearer café".encode())],
    )
    assert res.status_code == 401


def test_learning_endpoints(client, auth):
    # Status inicial
    res = client.get("/v1/learning/status", headers=auth)
    assert res.status_code == 200
    assert "cropTypes" in res.json()

    # Ingesta
    payload = {
        "readings": [
            {
                "cropId": "66f5a1000000000000000101",
                "cropType": "LETTUCE",
                "measuredAt": "2026-10-31T12:00:00Z",
                "localHour": 12,
                "measures": {
                    "temperature": 21.5,
                    "humidity": 62.0,
                    "brightness": 850.0,
                    "ph": 6.1,
                    "tds": 720.0,
                    "atmosphere": 1013.0,
                    "soilMoisture": 72.0,
                },
            }
        ]
    }
    res = client.post("/v1/learning/readings", json=payload, headers=auth)
    assert res.status_code == 200
    data = res.json()
    assert data["received"] == 1
    assert data["stored"] == 1

    # Olvido de cultivo
    res = client.delete("/v1/learning/crops/66f5a1000000000000000101", headers=auth)
    assert res.status_code == 200
    assert res.json()["deleted"] >= 1

    # Un identificador que no es ObjectId se rechaza antes de tocar el almacén.
    res = client.delete("/v1/learning/crops/crop-api-test", headers=auth)
    assert res.status_code == 422

    # Poda
    res = client.post("/v1/learning/prune", headers=auth)
    assert res.status_code == 200
    assert "removed" in res.json()


def test_validation_error(client, auth):
    # Tipo de cultivo inexistente
    payload = {
        "readings": [
            {
                "cropId": "crop-bad",
                "cropType": "BANANA_CROP_UNKNOWN",
                "measuredAt": "2026-10-31T12:00:00Z",
                "measures": {},
            }
        ]
    }
    res = client.post("/v1/learning/readings", json=payload, headers=auth)
    assert res.status_code == 422


def test_full_learning_cycle_through_the_api(isolated_client, auth, tmp_path):
    client = isolated_client
    crop_id = "66f5a1000000000000000101"
    readings = lettuce_series(crop_id, days=2.0, step_minutes=30.0)

    # Un lote con un duplicado: se acepta completo, pero la clave (cultivo, instante) evita la fila repetida.
    duplicate = readings[10]
    batch = payload_for([*readings, duplicate])
    res = client.post("/v1/learning/readings", json=batch, headers=auth)
    assert res.status_code == 200
    data = res.json()
    assert data["received"] == len(readings) + 1
    assert data["stored"] == len(readings)
    assert data["total"] == len(readings)

    # Reintentar el mismo lote no vuelve a almacenar nada.
    res = client.post("/v1/learning/readings", json=batch, headers=auth)
    assert res.json()["stored"] == 0
    assert res.json()["total"] == len(readings)

    # Un salto térmico dentro del rango físico se almacena, pero el filtro de derivada temporal lo descarta al entrenar.
    previous = readings[20]
    jump = replace(
        previous,
        measured_at=previous.measured_at + 30.0,
        values={**previous.values, "temperature": previous.values["temperature"] + 20.0},
    )
    res = client.post("/v1/learning/readings", json=payload_for([jump]), headers=auth)
    assert res.status_code == 200
    assert res.json()["stored"] == 1

    res = client.get("/v1/learning/status", headers=auth)
    assert res.status_code == 200
    assert res.json()["storedReadings"] == len(readings) + 1

    res = client.post("/v1/learning/train", json={"cropType": "lettuce"}, headers=auth)
    assert res.status_code == 200, res.text
    status = res.json()
    lettuce = next(item for item in status["cropTypes"] if item["cropType"] == "LETTUCE")
    assert lettuce["trainedAt"] is not None
    assert lettuce["quality"]["outliers"] >= 1
    assert lettuce["quality"]["score"] < 1.0

    # Exportación firmada: el manifiesto Ed25519/BLAKE2b verifica con la clave pública del servicio.
    inference = tmp_path / "models" / "LETTUCE_inference.json"
    manifest = tmp_path / "models" / "LETTUCE_manifest.json"
    bundle = json.loads(inference.read_text(encoding="utf-8"))
    assert bundle["schema"] == "smartpot.learned-models/1"
    assert bundle["crop_type"] == "LETTUCE"
    assert len(bundle["checks"]) == 16
    signature = json.loads(manifest.read_text(encoding="utf-8"))["signature_hex"]
    assert isinstance(signature, str) and signature
    trusted = public_key_from_pem(tmp_path / "keys" / "ed25519_public.pem")
    assert verify_manifest(manifest, trusted) is True

    # Un cultivo sin lecturas no entrena: 404 con el motivo, no un 500.
    res = client.post("/v1/learning/train", json={"cropType": "BASIL"}, headers=auth)
    assert res.status_code == 404
    assert res.json()["detail"] == "No readings for BASIL"

    # Olvido por cultivo y poda.
    res = client.delete(f"/v1/learning/crops/{crop_id}", headers=auth)
    assert res.status_code == 200
    assert res.json()["deleted"] == len(readings) + 1
    res = client.get("/v1/learning/status", headers=auth)
    lettuce = next(item for item in res.json()["cropTypes"] if item["cropType"] == "LETTUCE")
    assert lettuce["readings"] == 0
    res = client.post("/v1/learning/prune", headers=auth)
    assert res.status_code == 200
    assert res.json()["removed"] == 0


def test_validation_reports_nested_fields(isolated_client, auth):
    payload = {
        "readings": [
            {
                "cropId": "crop-invalid",
                "cropType": "LETTUCE",
                "measuredAt": "2026-11-02T12:00:00Z",
                "localHour": 24,
                "measures": {"temperature": 61, "humidity": -1},
            }
        ]
    }
    res = isolated_client.post("/v1/learning/readings", json=payload, headers=auth)
    assert res.status_code == 422
    body = res.json()
    assert body["detail"] == "Invalid data"
    fields = {error["field"] for error in body["errors"]}
    assert {"readings.0.localHour", "readings.0.measures.temperature", "readings.0.measures.humidity"} <= fields


def test_prune_invalidates_the_status_cache(tmp_path, auth, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "prune"))
    monkeypatch.setenv("LEARNING_RETENTION_DAYS", "1")
    monkeypatch.setenv("LEARNING_MIN_SAMPLES", "80")
    monkeypatch.setenv("LEARNING_CHECK_SECONDS", "0")
    monkeypatch.setenv("LEARNING_SEPARATE_PROCESS", "false")
    monkeypatch.setenv("SMARTPOT_AI_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        now = time.time()
        values = {
            "temperature": 20.0,
            "humidity": 60.0,
            "brightness": 800.0,
            "ph": 6.0,
            "tds": 700.0,
            "atmosphere": 1012.0,
            "soilMoisture": 70.0,
        }

        def reading(crop_id: str, moment: float) -> dict:
            return {
                "cropId": crop_id,
                "cropType": "LETTUCE",
                "measuredAt": datetime.fromtimestamp(moment, tz=UTC).isoformat(),
                "localHour": 12,
                "measures": values,
            }

        old = reading("66f5a1000000000000000201", now - 10 * 86_400)
        fresh = reading("66f5a1000000000000000202", now)
        res = client.post("/v1/learning/readings", json={"readings": [old, fresh]}, headers=auth)
        assert res.json()["stored"] == 2
        assert client.get("/v1/learning/status", headers=auth).json()["cropTypes"][0]["readings"] == 2

        res = client.post("/v1/learning/prune", headers=auth)
        assert res.json()["removed"] == 1
        # Sin invalidar la caché, el status seguiría informando 2 lecturas durante 30 s.
        lettuce = client.get("/v1/learning/status", headers=auth).json()["cropTypes"][0]
        assert lettuce["readings"] == 1
        get_settings.cache_clear()


def test_unknown_routes_and_methods(isolated_client, auth):
    res = isolated_client.get("/v1/learning/unknown", headers=auth)
    assert res.status_code == 404
    res = isolated_client.put("/v1/learning/prune", headers=auth)
    assert res.status_code == 405
