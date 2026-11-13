"""Data quality assessment prior to model training.

- Completeness: fraction of present values.
- Validity: fraction of values within the physical sensor scale.
- Interquartile range (IQR with k = 3) outliers.
- Physically impossible per-minute jumps for temperature, humidity and substrate.
- An Isolation Forest secondary filter when timestamps are available, so stealthy poison rows
  that pass the IQR test do not reach the training set.

The Data Quality Score (DQS) is the product of completeness, validity, and the fraction of usable rows.
"""

from dataclasses import dataclass

import numpy as np

try:
    import polars as pl
except Exception:
    pl = None  # type: ignore[assignment]

from sklearn.ensemble import IsolationForest

from app.learning.store import COLUMNS

PHYSICAL = {
    "temperature": (-20, 60),
    "humidity": (0, 100),
    "brightness": (0, 200_000),
    "ph": (0, 14),
    "tds": (0, 10_000),
    "atmosphere": (300, 1_100),
    "soil_moisture": (0, 100),
}
IQR_K = 3.0
SEED = 7
# Cambio físico máximo razonable por minuto; un salto mayor es una lectura corrupta o inyectada.
MAX_RATE_PER_MINUTE = {
    "temperature": 1.0,
    "humidity": 5.0,
    "soil_moisture": 2.0,
}
FOREST_MIN_ROWS = 100
FOREST_CONTAMINATION = 0.005


@dataclass(frozen=True)
class QualityReport:
    rows: int
    completeness: float
    validity: float
    outliers: int
    score: float
    keep: np.ndarray


def _finalize(
    values: np.ndarray,
    present: np.ndarray,
    valid_rows: np.ndarray,
    outlier: np.ndarray,
    times: np.ndarray | None,
    crops: np.ndarray | None,
    validity: float,
) -> QualityReport:
    rows = values.shape[0]
    if rows == 0:
        return QualityReport(0, 0.0, 0.0, 0, 0.0, np.zeros(0, dtype=bool))

    if times is not None:
        outlier = outlier | _temporal_outliers(times, values, present, crops)
        outlier = outlier | _forest_outliers(values, present, valid_rows & ~outlier)

    keep = valid_rows & ~outlier
    completeness = float(present.mean())
    score = completeness * validity * float(keep.mean()) if rows else 0.0
    return QualityReport(
        rows=rows,
        completeness=round(completeness, 4),
        validity=round(validity, 4),
        outliers=int(outlier.sum()),
        score=round(score, 4),
        keep=keep,
    )


def _validity(values: np.ndarray, present: np.ndarray) -> float:
    valid = np.ones_like(present)
    for i, name in enumerate(COLUMNS):
        low, high = PHYSICAL[name]
        with np.errstate(invalid="ignore"):
            valid[:, i] = ~present[:, i] | ((values[:, i] >= low) & (values[:, i] <= high))
    return float(valid[present].mean()) if present.any() else 0.0


def _temporal_outliers(
    times: np.ndarray,
    values: np.ndarray,
    present: np.ndarray,
    crops: np.ndarray | None,
) -> np.ndarray:
    """Flags a reading when its per-minute change exceeds the physical limit for its crop."""
    flagged = np.zeros(values.shape[0], dtype=bool)
    groups = np.unique(crops) if crops is not None else np.array([None], dtype=object)
    for group in groups:
        rows = np.flatnonzero(crops == group) if crops is not None else np.arange(values.shape[0])
        if rows.size < 2:
            continue
        order = rows[np.argsort(times[rows], kind="stable")]
        for name, limit in MAX_RATE_PER_MINUTE.items():
            column = COLUMNS.index(name)
            previous: int | None = None
            for row in order:
                if not present[row, column] or not np.isfinite(values[row, column]):
                    continue
                if previous is not None:
                    elapsed = (times[row] - times[previous]) / 60.0
                    if elapsed > 0.0:
                        rate = abs(values[row, column] - values[previous, column]) / max(elapsed, 1.0)
                        if rate > limit:
                            flagged[row] = True
                previous = row
    return flagged


def _forest_outliers(
    values: np.ndarray,
    present: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    """Secondary unsupervised filter over the raw sensor columns; requires timelines."""
    flagged = np.zeros(values.shape[0], dtype=bool)
    rows = np.flatnonzero(keep)
    if rows.size < FOREST_MIN_ROWS:
        return flagged
    matrix = np.empty((rows.size, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        available = values[rows, column][present[rows, column]]
        fill = float(np.median(available)) if available.size else 0.0
        matrix[:, column] = np.where(present[rows, column], values[rows, column], fill)
    if not np.isfinite(matrix).all():
        return flagged
    scale = matrix.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    normalized = (matrix - matrix.mean(axis=0)) / scale
    forest = IsolationForest(
        n_estimators=100,
        contamination=FOREST_CONTAMINATION,
        random_state=SEED,
        n_jobs=1,
    ).fit(normalized)
    predictions = forest.predict(normalized)
    flagged[rows[predictions == -1]] = True
    return flagged


def assess_polars(
    values: np.ndarray, times: np.ndarray | None = None, crops: np.ndarray | None = None
) -> QualityReport:
    """Data quality evaluation accelerated in parallel using Polars LazyFrame."""
    rows = values.shape[0]
    if rows == 0:
        return QualityReport(0, 0.0, 0.0, 0, 0.0, np.zeros(0, dtype=bool))

    df = pl.DataFrame(values, schema=list(COLUMNS)).fill_nan(None)
    present = ~np.isnan(values)

    valid_exprs = []
    outlier_exprs = []
    for name in COLUMNS:
        low, high = PHYSICAL[name]
        column = pl.col(name)
        valid_exprs.append(column.is_null() | ((column >= low) & (column <= high)))

        q1 = column.quantile(0.25, interpolation="linear")
        q3 = column.quantile(0.75, interpolation="linear")
        spread = pl.max_horizontal(q3 - q1, pl.lit(0.02 * (high - low)))
        outlier_exprs.append(
            (
                (column.count() >= 20)
                & column.is_not_null()
                & ((column < q1 - IQR_K * spread) | (column > q3 + IQR_K * spread))
            ).fill_null(False)
        )

    eval_df = (
        df.lazy()
        .select(
            [
                pl.all_horizontal(valid_exprs).alias("all_valid"),
                pl.any_horizontal(outlier_exprs).alias("any_outlier"),
            ]
        )
        .collect()
    )

    valid_rows = eval_df["all_valid"].to_numpy()
    outlier = eval_df["any_outlier"].to_numpy()
    return _finalize(values, present, valid_rows, outlier, times, crops, _validity(values, present))


def assess(values: np.ndarray, times: np.ndarray | None = None, crops: np.ndarray | None = None) -> QualityReport:
    """Evaluates reading quality using Polars with graceful fallback to pure NumPy."""
    rows = values.shape[0]
    if rows == 0:
        return QualityReport(0, 0.0, 0.0, 0, 0.0, np.zeros(0, dtype=bool))

    if pl is not None:
        try:
            return assess_polars(values, times, crops)
        except Exception:
            pass

    present = ~np.isnan(values)
    valid = np.ones_like(present)
    for i, name in enumerate(COLUMNS):
        low, high = PHYSICAL[name]
        with np.errstate(invalid="ignore"):
            valid[:, i] = ~present[:, i] | ((values[:, i] >= low) & (values[:, i] <= high))
    valid_rows = valid.all(axis=1)
    validity = float(valid[present].mean()) if present.any() else 0.0

    outlier = np.zeros(rows, dtype=bool)
    for i in range(values.shape[1]):
        data = values[present[:, i], i]
        if data.size < 20:
            continue
        q1, q3 = np.percentile(data, [25, 75])
        low, high = PHYSICAL[COLUMNS[i]]
        spread = max(q3 - q1, 0.02 * (high - low))
        with np.errstate(invalid="ignore"):
            outlier |= present[:, i] & ((values[:, i] < q1 - IQR_K * spread) | (values[:, i] > q3 + IQR_K * spread))
    return _finalize(values, present, valid_rows, outlier, times, crops, validity)
