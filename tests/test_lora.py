import numpy as np

from app.learning.lora import LowRankCropAdapter


def test_lora_initialization():
    adapter = LowRankCropAdapter(d_in=10, k_out=1, rank=2, alpha=1.0)
    # Inicialmente B es cero, por lo que W = W0 exactamente
    np.testing.assert_array_equal(adapter.effective_weight, adapter.w0)
    assert adapter.updates_count == 0


def test_lora_predict_proba():
    adapter = LowRankCropAdapter(d_in=5, k_out=1, rank=2)
    x = np.random.randn(20, 5)
    proba = adapter.predict_proba(x)
    assert proba.shape == (20, 2)
    # Las probabilidades deben sumar 1 por fila y estar entre 0 y 1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), atol=1e-5)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_lora_online_update():
    rng = np.random.default_rng(42)
    adapter = LowRankCropAdapter(d_in=6, k_out=1, rank=2)
    x_batch = rng.normal(0, 1, size=(30, 6))
    y_batch = (x_batch[:, 0] > 0.0).astype(float)

    # Pérdida inicial
    loss_init = adapter.online_update(x_batch, y_batch, lr=0.1)
    assert adapter.updates_count == 1
    assert adapter.last_update_ms > 0.0

    # Iterar varias actualizaciones: la pérdida debe descender
    loss = loss_init
    for _ in range(15):
        loss = adapter.online_update(x_batch, y_batch, lr=0.1)
    assert loss < loss_init
    # B ya no es puramente cero
    assert not np.all(adapter.b == 0.0)


def test_lora_serialization():
    adapter = LowRankCropAdapter(d_in=8, k_out=1, rank=3, alpha=2.0)
    x = np.random.randn(10, 8)
    y = np.ones(10)
    adapter.online_update(x, y, lr=0.05)

    data = adapter.to_dict()
    restored = LowRankCropAdapter.from_dict(data)

    assert restored.d_in == adapter.d_in
    assert restored.rank == adapter.rank
    assert restored.updates_count == adapter.updates_count
    np.testing.assert_allclose(restored.effective_weight, adapter.effective_weight)
    np.testing.assert_allclose(restored.predict_proba(x), adapter.predict_proba(x))
