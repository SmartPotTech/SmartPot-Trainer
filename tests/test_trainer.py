import numpy as np

from app.knowledge.profiles import get_profile
from app.learning.features import build
from app.learning.store import ReadingStore
from app.learning.trainer import (
    train_all,
    train_anomaly,
    train_classifier,
    train_regressor,
    train_states,
)
from tests.learning_data import lettuce_series


def test_train_insufficient_samples():
    card = train_classifier("needs_water", np.empty((10, 10)), np.empty(10), min_samples=200, current=None)
    assert card.status == "PENDING"
    assert "Insufficient readings" in card.reason


def test_train_classifier_with_a_single_class_stays_pending():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 10))
    y = np.ones(300)
    card = train_classifier("needs_water", x, y, min_samples=100, current=None)
    assert card.status == "PENDING"
    assert "one class" in card.reason.lower()
    assert card.estimator is None


def test_train_states():
    rng = np.random.default_rng(42)
    # 100 lecturas en dos grupos claros
    cluster_1 = rng.normal(0.2, 0.05, size=(50, 6))
    cluster_2 = rng.normal(0.8, 0.05, size=(50, 6))
    positions = np.vstack([cluster_1, cluster_2])
    hours = np.hstack([np.full(50, 9.0), np.full(50, 15.0)])

    states = train_states(positions, hours)
    assert states is not None
    assert states.k >= 2
    assert states.silhouette > 0.3
    assert len(states.clusters) >= 2


def test_train_anomaly():
    rng = np.random.default_rng(42)
    positions = rng.normal(0.5, 0.1, size=(120, 6))
    anomaly = train_anomaly(positions)
    assert anomaly is not None
    assert anomaly.samples == 120

    # Lectura normal vs anómala extrema
    prob_norm, is_anom_norm = anomaly.score(np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]))
    prob_ext, is_anom_ext = anomaly.score(np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]))
    assert prob_ext > prob_norm


def test_train_all_with_synthetic_series():
    profile = get_profile("LETTUCE")
    store = ReadingStore(None)
    # Generar 4 días de lecturas de lechuga para tener suficientes eventos
    store.add(lettuce_series("crop-train", days=4.0, step_minutes=4.0))
    series = store.load("LETTUCE", limit=2000)
    dataset = build(profile, series)

    models = train_all(dataset, min_samples=100, current={})
    assert "needs_water" in models
    assert "overheat" in models
    assert "moisture_1h" in models
    assert "states" in models
    assert "anomaly" in models

    # Comprobar que los modelos no supervisados se entrenan
    assert models["states"] is not None
    assert models["anomaly"] is not None
    store.close()


def test_train_regressor_baseline_uses_mean_not_zero():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 10))
    # Media constante cercana a 25.0 (no 0): un baseline nulo (zeros_like) daría MAE ~25.
    y = 25.0 + 0.05 * x[:, 0] + rng.normal(scale=0.01, size=80)
    card = train_regressor(x, y, min_samples=40, current=None)
    assert card.baseline is not None
    assert card.baseline < 1.0
