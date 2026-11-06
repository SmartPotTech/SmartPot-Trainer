import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from app.learning.service import TrainerConfig, TrainerService
from tests.learning_data import lettuce_series


def public_key_from_pem(path: Path) -> str:
    key = serialization.load_pem_public_key(path.read_bytes())
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def test_service_lifecycle_and_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = TrainerConfig(
            data_dir=tmpdir,
            min_samples=100,
            retrain_every=50,
            check_seconds=0,
            separate_process=False,  # en pruebas unitarias usamos el mismo proceso
            smartpot_ai_url="http://127.0.0.1:9999",  # dummy URL
        )
        service = TrainerService(base)
        service.start()

        # Ingerir lecturas
        readings = lettuce_series("crop-srv", days=3.0)
        stored = service.ingest(readings)
        assert stored == len(readings)

        # Verificar due
        due_crops = service.due()
        assert "LETTUCE" in due_crops

        # Entrenar
        state = service.train("LETTUCE")
        assert state is not None
        assert state.quality["rows"] > 0
        assert (Path(tmpdir) / "models" / "LETTUCE_inference.json").exists()
        assert (Path(tmpdir) / "keys" / "ed25519_private.pem").exists()

        # Verificar status
        status = service.status()
        assert status["enabled"]
        assert status["persistent"]
        assert status["stored_readings"] == len(readings)
        lettuce_info = next(c for c in status["crop_types"] if c["crop_type"] == "LETTUCE")
        assert lettuce_info["readings"] == len(readings)
        assert lettuce_info["trained_at"] is not None

        # Verificar persistencia en disco
        service.stop()

        # Solo se recarga un joblib autenticado con la clave pública fijada.
        trusted = public_key_from_pem(Path(tmpdir) / "keys" / "ed25519_public.pem")
        verified = TrainerConfig(
            data_dir=tmpdir,
            min_samples=100,
            retrain_every=50,
            check_seconds=0,
            separate_process=False,
            smartpot_ai_url="http://127.0.0.1:9999",
            trusted_model_public_key_hex=trusted,
        )
        service2 = TrainerService(verified)
        assert "LETTUCE" in service2.types
        assert service2.types["LETTUCE"].quality["rows"] > 0
        service2.stop()


def test_unverified_joblib_is_never_deserialized():
    with tempfile.TemporaryDirectory() as tmpdir:
        models = Path(tmpdir) / "models"
        models.mkdir(parents=True)
        marker = Path(tmpdir) / "pwned.txt"
        payload = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('executed')\n"
        evil = type("Evil", (), {"__reduce__": lambda self: (exec, (payload,))})()
        import joblib

        joblib.dump({"sklearn": "1.9.1", "state": evil}, models / "LETTUCE.joblib")
        config = TrainerConfig(
            data_dir=tmpdir,
            min_samples=100,
            retrain_every=50,
            check_seconds=0,
            separate_process=False,
            smartpot_ai_url="http://127.0.0.1:9999",
            trusted_model_public_key_hex="0" * 64,
        )
        service = TrainerService(config)
        assert "LETTUCE" not in service.types
        assert not marker.exists()
        service.stop()
