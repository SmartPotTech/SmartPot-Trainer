import tempfile

import pytest

from app.learning.store import ReadingStore
from tests.learning_data import lettuce_series


def test_reading_store_memory():
    store = ReadingStore(None)
    assert not store.persistent
    assert store.total() == 0

    readings = lettuce_series("pot-1", days=1.0)
    added = store.add(readings)
    assert added == len(readings)
    assert store.total() == len(readings)

    # Duplicados son ignorados
    added_dups = store.add(readings)
    assert added_dups == 0
    assert store.total() == len(readings)

    counts = store.counts()
    assert "LETTUCE" in counts
    assert counts["LETTUCE"]["readings"] == len(readings)
    assert counts["LETTUCE"]["crops"] == 1

    df = store.load_df("LETTUCE", limit=100)
    assert len(df) == 100
    assert "crop_key" in df.columns
    assert "temperature" in df.columns

    series = store.load("LETTUCE", limit=50)
    assert len(series) == 50
    assert series.values.shape == (50, 7)

    # Olvido de cultivo
    deleted = store.forget("pot-1")
    assert deleted == len(readings)
    assert store.total() == 0
    store.close()


def test_reading_store_persistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ReadingStore(tmpdir)
        assert store.persistent

        r1 = lettuce_series("pot-a", days=0.5, start=1_000_000.0)
        r2 = lettuce_series("pot-b", days=0.5, start=2_000_000.0)
        store.add(r1)
        store.add(r2)

        assert store.total() == len(r1) + len(r2)
        counts = store.counts()
        assert counts["LETTUCE"]["crops"] == 2

        # Poda por tope
        removed = store.prune(retention_days=365, max_rows_per_type=len(r1))
        assert removed > 0
        assert store.total() <= len(r1)
        store.close()


def test_timezone_rejects_sql_injection():
    from app.learning.store import _checked_timezone

    with pytest.raises(ValueError):
        _checked_timezone("America/Bogota'; DROP TABLE crops; --")


def test_unknown_crop_type_never_reaches_timescale():
    store = ReadingStore(None, database_url="postgresql://invalid", timezone="UTC")
    assert store.load_df("LETTUCE' OR '1'='1", 10).is_empty()


def test_species_only_in_the_local_store_still_count_and_train(monkeypatch):
    """Las lecturas ingeridas para una especie ausente de TimescaleDB deben contar y entrenar."""
    from dataclasses import replace

    import polars as pl

    store = ReadingStore(None, database_url="postgresql://invalid", timezone="UTC")
    pepper = [replace(item, crop_type="PEPPER") for item in lettuce_series("pot-pepper", days=1.0)]
    store.add(pepper)

    def fake_adbc(query: str):
        # La consulta de conteos ve LETTUCE en la base; la carga de PEPPER no tiene filas
        # y debe caer al almacén local.
        if "GROUP BY c.type" in query:
            return pl.DataFrame(
                {"crop_type": ["LETTUCE"], "readings": [10], "crops": [1], "last_received": [1_700_000_000.0]}
            )
        return pl.DataFrame()

    monkeypatch.setattr(store, "_adbc", fake_adbc)
    counts = store.counts()
    assert counts["LETTUCE"]["readings"] == 10
    assert counts["PEPPER"]["readings"] == len(pepper)
    assert not store.load_df("PEPPER", limit=50).is_empty()
    store.close()


def test_load_samples_fairly_across_crops_without_starvation(monkeypatch):
    from datetime import UTC, datetime

    import polars as pl

    from app.learning.store import StoredReading, crop_key

    store = ReadingStore(None)
    base_ts = datetime(2026, 11, 1, tzinfo=UTC).timestamp()
    # Un cultivo muy ruidoso publica más reciente que otro cultivo estable; sin balanceo
    # uno solo llenaría todo el límite y el otro quedaría sin representación.
    steady = [
        StoredReading("crop-steady", "LETTUCE", base_ts + i * 60.0, 10, {"temperature": 21.0})
        for i in range(20)
    ]
    noisy = [
        StoredReading("crop-noisy", "LETTUCE", base_ts + 5_000.0 + i * 10.0, 11, {"temperature": 22.0})
        for i in range(60)
    ]
    assert store.add(steady + noisy) == 80

    df = store.load_df("LETTUCE", limit=20)
    assert df.height == 20
    keys = set(df["crop_key"].to_list())
    assert keys == {crop_key("crop-steady"), crop_key("crop-noisy")}
    assert df.filter(pl.col("crop_key") == crop_key("crop-steady")).height == 10
    assert df.filter(pl.col("crop_key") == crop_key("crop-noisy")).height == 10

    series = store.load("LETTUCE", limit=20)
    assert len(series) == 20
    assert set(series.crops.tolist()) == {crop_key("crop-steady"), crop_key("crop-noisy")}

    # Verifica también la ruta TimescaleDB/ADBC: usa CROSS JOIN LATERAL balanceado y sin GROUP BY c.type.
    captured_queries: list[str] = []
    remote_store = ReadingStore(None, database_url="postgresql://invalid", timezone="UTC")

    def fake_adbc(query: str):
        captured_queries.append(query)
        return None

    monkeypatch.setattr(remote_store, "_adbc", fake_adbc)
    remote_store.load_df("LETTUCE", limit=20)
    assert len(captured_queries) == 1
    assert "CROSS JOIN LATERAL" in captured_queries[0]
    assert "GROUP BY c.type" not in captured_queries[0]
    store.close()
    remote_store.close()

