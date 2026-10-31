import numpy as np

from app.learning.quality import assess, assess_polars
from app.learning.store import COLUMNS


def test_quality_empty():
    report = assess(np.empty((0, len(COLUMNS))))
    assert report.rows == 0
    assert report.completeness == 0.0
    assert report.score == 0.0


def test_quality_clean_data():
    rng = np.random.default_rng(42)
    n = 200
    # Valores dentro de rangos normales
    values = np.column_stack(
        [
            rng.uniform(18.0, 24.0, n),  # temperature
            rng.uniform(50.0, 70.0, n),  # humidity
            rng.uniform(400.0, 1200.0, n),  # brightness
            rng.uniform(5.8, 6.5, n),  # ph
            rng.uniform(600.0, 800.0, n),  # tds
            rng.uniform(1010.0, 1015.0, n),  # atmosphere
            rng.uniform(65.0, 75.0, n),  # soil_moisture
        ]
    )
    report = assess(values)
    assert report.rows == n
    assert report.completeness == 1.0
    assert report.validity == 1.0
    assert report.outliers == 0
    assert report.score > 0.95


def test_quality_outliers_and_invalid():
    rng = np.random.default_rng(42)
    n = 100
    values = np.column_stack(
        [
            rng.uniform(20.0, 22.0, n),
            rng.uniform(60.0, 65.0, n),
            rng.uniform(500.0, 600.0, n),
            rng.uniform(6.0, 6.2, n),
            rng.uniform(700.0, 720.0, n),
            rng.uniform(1012.0, 1014.0, n),
            rng.uniform(70.0, 72.0, n),
        ]
    )
    # Insertar valores imposibles físicamente
    values[0, 0] = 999.0  # temp imposible
    values[1, 3] = -10.0  # ph imposible
    # Insertar atípicos estadísticos (IQR)
    values[2, 2] = 20_000.0

    report = assess(values)
    assert report.rows == n
    assert report.outliers >= 1
    assert not report.keep[0]  # valor físico inválido excluido
    assert not report.keep[1]
    assert not report.keep[2]


def test_quality_parity_polars_and_fallback():
    rng = np.random.default_rng(123)
    n = 300
    values = np.column_stack(
        [
            rng.uniform(15.0, 25.0, n),
            rng.uniform(40.0, 80.0, n),
            rng.uniform(200.0, 1500.0, n),
            rng.uniform(5.5, 6.8, n),
            rng.uniform(500.0, 900.0, n),
            rng.uniform(1005.0, 1020.0, n),
            rng.uniform(55.0, 85.0, n),
        ]
    )
    # Atípico intencional
    values[50, 4] = 9500.0

    rep_polars = assess_polars(values)
    rep_general = assess(values)

    assert rep_polars.rows == rep_general.rows
    assert rep_polars.outliers == rep_general.outliers
    assert abs(rep_polars.score - rep_general.score) < 1e-4
    np.testing.assert_array_equal(rep_polars.keep, rep_general.keep)
