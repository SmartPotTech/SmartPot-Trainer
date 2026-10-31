import tempfile

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
