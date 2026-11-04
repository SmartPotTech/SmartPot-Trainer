"""Data quality assessment prior to model training.

- Completeness: fraction of present values.
- Validity: fraction of values within the physical sensor scale.
- Interquartile range (IQR with k = 3) outliers: excluded from training so a faulty sensor
  does not teach spurious patterns.

The Data Quality Score (DQS) is the product of completeness, validity, and the fraction of usable rows.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

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


@dataclass(frozen=True)
class QualityReport:
    rows: int
    completeness: float
    validity: float
    outliers: int
    score: float
    keep: np.ndarray


def assess_polars(values: np.ndarray) -> QualityReport:
    """Data quality evaluation accelerated in parallel using Polars LazyFrame."""
    rows = values.shape[0]
    if rows == 0:
        return QualityReport(0, 0.0, 0.0, 0, 0.0, np.zeros(0, dtype=bool))

    df = pl.DataFrame(values, schema=list(COLUMNS))
    present = ~np.isnan(values)
    completeness = float(present.mean())

    valid_exprs = []
    outlier_exprs = []

    for name in COLUMNS:
        low, high = PHYSICAL[name]
        c = pl.col(name)
        valid_exprs.append(c.is_null() | ((c >= low) & (c <= high)))

        q1 = c.quantile(0.25, interpolation="linear")
        q3 = c.quantile(0.75, interpolation="linear")
        spread = pl.max_horizontal(q3 - q1, pl.lit(0.02 * (high - low)))
        outlier_exprs.append(c.is_not_null() & ((c < q1 - IQR_K * spread) | (c > q3 + IQR_K * spread)))

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

    valid_mask = eval_df["all_valid"].to_numpy()
    outlier_mask = eval_df["any_outlier"].to_numpy()
    keep = valid_mask & ~outlier_mask

    valid = np.ones_like(present)
    for i, name in enumerate(COLUMNS):
        low, high = PHYSICAL[name]
        with np.errstate(invalid="ignore"):
            valid[:, i] = ~present[:, i] | ((values[:, i] >= low) & (values[:, i] <= high))
    validity = float(valid[present].mean()) if present.any() else 0.0
    score = completeness * validity * float(keep.mean())

    return QualityReport(
        rows=rows,
        completeness=round(completeness, 4),
        validity=round(validity, 4),
        outliers=int(outlier_mask.sum()),
        score=round(score, 4),
        keep=keep,
    )


def assess(values: np.ndarray) -> QualityReport:
    """Evaluates reading quality using Polars with graceful fallback to pure NumPy."""
    rows = values.shape[0]
    if rows == 0:
        return QualityReport(0, 0.0, 0.0, 0, 0.0, np.zeros(0, dtype=bool))

    try:
        return assess_polars(values)
    except Exception:
        pass

    present = ~np.isnan(values)
    completeness = float(present.mean())
    valid = np.ones_like(present)
    for i, name in enumerate(COLUMNS):
        low, high = PHYSICAL[name]
        with np.errstate(invalid="ignore"):
            valid[:, i] = ~present[:, i] | ((values[:, i] >= low) & (values[:, i] <= high))
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
    keep = ~outlier & valid.all(axis=1)
    score = completeness * validity * float(keep.mean())
    return QualityReport(
        rows=rows,
        completeness=round(completeness, 4),
        validity=round(validity, 4),
        outliers=int(outlier.sum()),
        score=round(score, 4),
        keep=keep,
    )
