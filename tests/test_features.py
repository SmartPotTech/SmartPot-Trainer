import numpy as np

from app.knowledge.profiles import get_profile
from app.learning.features import (
    _hour_features,
    _slopes,
    build,
    camel_columns,
    live_features,
    positions,
    to_store,
)
from app.learning.store import ReadingStore
from tests.learning_data import lettuce_series


def test_positions():
    profile = get_profile("LETTUCE")
    # Para LETTUCE: temperature es (15, 22). Si t=15 -> 0, si t=22 -> 1, si t=18.5 -> 0.5
    values = np.array([[18.5, 60.0, 850.0, 6.0, 700.0, 1012.0, 70.0]])
    pos = positions(profile, values)
    assert pos.shape == (1, 6)
    assert abs(pos[0, 0] - 0.5) < 1e-4


def test_hour_features():
    hours = np.array([0.0, 6.0, 12.0, 18.0])
    times = hours * 3600.0
    hf = _hour_features(hours, times)
    assert hf.shape == (4, 2)
    # A las 6h, el ángulo es 6/24 * 2pi = pi/2 -> sin=1, cos=0
    assert abs(hf[1, 0] - 1.0) < 1e-4
    assert abs(hf[1, 1] - 0.0) < 1e-4


def test_slopes():
    times = np.array([0.0, 600.0, 1200.0, 1800.0])
    series = np.array([10.0, 12.0, 14.0, 16.0])
    slopes = _slopes(times, series, scale=10.0)
    assert len(slopes) == 4
    # En 1800 s subió de 10 a 16 (+6). Ventana de media hora = 0.5h. Tasa = +12 unidades/h. Normalizado / 10 = 1.2
    assert slopes[-1] > 0.0


def test_build_dataset():
    profile = get_profile("LETTUCE")
    store = ReadingStore(None)
    store.add(lettuce_series("crop-1", days=2.0))
    series = store.load("LETTUCE", limit=1000)
    dataset = build(profile, series)

    assert len(dataset) > 0
    assert dataset.x.shape[1] == 10  # 6 pos + 2 hour + 2 slopes
    assert dataset.water_mask.shape[0] == len(dataset)
    assert dataset.heat_mask.shape[0] == len(dataset)
    store.close()


def test_live_features():
    profile = get_profile("LETTUCE")
    measures = {
        "temperature": 20.0,
        "humidity": 60.0,
        "brightness": 800.0,
        "ph": 6.0,
        "tds": 700.0,
        "soilMoisture": 70.0,
    }
    history = [(1000.0, measures), (2000.0, measures)]
    features = live_features(profile, measures, hour=14, moment=3000.0, history=history)
    assert features is not None
    assert features.shape == (1, 10)


def test_conversions():
    cols = camel_columns()
    assert "soilMoisture" in cols
    st = to_store({"soilMoisture": 75.0, "temperature": 21.0})
    assert "soil_moisture" in st
    assert st["soil_moisture"] == 75.0
