import tempfile

from app.learning.service import TrainerConfig, TrainerService
from tests.learning_data import lettuce_series


def test_service_lifecycle_and_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            data_dir=tmpdir,
            min_samples=100,
            retrain_every=50,
            check_seconds=0,
            separate_process=False,  # en pruebas unitarias usamos el mismo proceso
            smartpot_ai_url="http://127.0.0.1:9999",  # dummy URL
        )
        service = TrainerService(config)
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

        # Al crear un nuevo servicio sobre el mismo directorio, carga los modelos
        service2 = TrainerService(config)
        assert "LETTUCE" in service2.types
        assert service2.types["LETTUCE"].quality["rows"] > 0
        service2.stop()
