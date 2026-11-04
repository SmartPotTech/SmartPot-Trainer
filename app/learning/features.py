"""Input features and self-supervised labels for learning from real readings.

Each reading is expressed as its normalized position within the species ideal range (0 = min, 1 = max),
the circular hour of the day, and the recent substrate and temperature slope. This allows one model
to generalize across all crops of the same species.

Labels are derived from the data itself (self-supervised learning): what happened in the hour following
each reading determines whether the crop ended up needing irrigation or ventilation.
"""

import math
from dataclasses import dataclass

import numpy as np

from app.knowledge.profiles import PARAMETERS, CropProfile
from app.learning.store import COLUMNS, Series

HORIZON_SECONDS = 3600.0
COVERAGE_SECONDS = 3000.0
SLOPE_WINDOW_SECONDS = 1800.0
MIN_SLOPE_SPAN_SECONDS = 600.0
TARGET_TOLERANCE_SECONDS = 600.0

STORE_NAMES = {"soilMoisture": "soil_moisture"}
INDEX = {column: i for i, column in enumerate(COLUMNS)}
FEATURE_NAMES = (
    *(f"position_{p}" for p in PARAMETERS),
    "hour_sin",
    "hour_cos",
    "substrate_trend",
    "temperature_trend",
)
STATE_PARAMETERS = PARAMETERS


def column(parameter: str) -> int:
    return INDEX[STORE_NAMES.get(parameter, parameter)]


def positions(profile: CropProfile, values: np.ndarray) -> np.ndarray:
    """Returns an n × 6 matrix with the position of each variable within its ideal range."""
    result = np.empty((values.shape[0], len(PARAMETERS)))
    for i, parameter in enumerate(PARAMETERS):
        rng = profile.ranges[parameter]
        result[:, i] = (values[:, column(parameter)] - rng.min) / (rng.max - rng.min)
    return result


def _hour_features(hours: np.ndarray, times: np.ndarray) -> np.ndarray:
    fallback = (times / 3600.0) % 24
    hour = np.where(np.isnan(hours), fallback, hours)
    angle = hour / 24.0 * 2 * math.pi
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _slopes(times: np.ndarray, series: np.ndarray, scale: float) -> np.ndarray:
    """Computes hourly change over the last half hour, normalized by ideal range width; 0 if history is insufficient."""
    n = times.shape[0]
    if n == 0:
        return np.zeros(0)
    starts = np.searchsorted(times, times - SLOPE_WINDOW_SECONDS, side="left")
    spans = times - times[starts]
    valid = (spans >= MIN_SLOPE_SPAN_SECONDS) & ~np.isnan(series) & ~np.isnan(series[starts])
    slopes = np.zeros(n)
    slopes[valid] = (series[valid] - series[starts[valid]]) / (scale * (spans[valid] / 3600.0))
    return slopes


@dataclass
class Dataset:
    """Rows ordered in global chronological order (required by time-series cross-validation)."""

    x: np.ndarray
    times: np.ndarray
    water: np.ndarray
    water_mask: np.ndarray
    heat: np.ndarray
    heat_mask: np.ndarray
    moisture_delta: np.ndarray
    states: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])


def build(profile: CropProfile, series: Series) -> Dataset:
    soil = profile.ranges["soilMoisture"]
    temperature = profile.ranges["temperature"]
    blocks = []
    for crop in np.unique(series.crops):
        rows = np.flatnonzero(series.crops == crop)
        times = series.times[rows]
        values = series.values[rows]
        pos = positions(profile, values)
        soil_values = values[:, column("soilMoisture")]
        temp_values = values[:, column("temperature")]
        features = np.column_stack(
            [
                pos,
                _hour_features(series.hours[rows], times),
                _slopes(times, soil_values, soil.max - soil.min),
                _slopes(times, temp_values, temperature.max - temperature.min),
            ]
        )

        n = times.shape[0]
        water = np.zeros(n)
        heat = np.zeros(n)
        covered = np.zeros(n, dtype=bool)
        delta = np.full(n, np.nan)
        ends = np.searchsorted(times, times + HORIZON_SECONDS, side="right")
        for i in range(n):
            window = slice(i + 1, ends[i])
            if ends[i] <= i + 1 or times[ends[i] - 1] - times[i] < COVERAGE_SECONDS:
                continue
            covered[i] = True
            future_soil = soil_values[window]
            future_temp = temp_values[window]
            water[i] = float(np.nanmin(future_soil, initial=np.inf) < soil.min)
            heat[i] = float(np.nanmax(future_temp, initial=-np.inf) > temperature.max)
            target = int(np.argmin(np.abs(times[window] - (times[i] + HORIZON_SECONDS))))
            if abs(times[i + 1 + target] - times[i] - HORIZON_SECONDS) <= TARGET_TOLERANCE_SECONDS:
                delta[i] = soil_values[i + 1 + target] - soil_values[i]

        complete = ~np.isnan(pos).any(axis=1)
        water_mask = covered & complete & (pos[:, PARAMETERS.index("soilMoisture")] >= 0)
        heat_mask = covered & complete & (pos[:, PARAMETERS.index("temperature")] <= 1)
        blocks.append((features, times, water, water_mask, heat, heat_mask, np.where(complete, delta, np.nan), pos))

    if not blocks:
        empty = np.empty(0)
        return Dataset(
            np.empty((0, len(FEATURE_NAMES))),
            empty,
            empty,
            empty.astype(bool),
            empty,
            empty.astype(bool),
            empty,
            np.empty((0, len(PARAMETERS))),
        )

    stacked = [np.concatenate([block[k] for block in blocks]) for k in range(8)]
    order = np.argsort(stacked[1], kind="stable")
    return Dataset(*(array[order] for array in stacked))


def live_features(
    profile: CropProfile,
    measures: dict[str, float | None],
    hour: int | None,
    moment: float,
    history: list[tuple[float, dict[str, float | None]]],
) -> np.ndarray | None:
    """Computes the same feature vector for the current live reading using its history (epoch seconds, measures)."""
    values = np.array([[measures.get(p) if measures.get(p) is not None else np.nan for p in camel_columns()]])
    pos = positions(profile, values)
    if np.isnan(pos).any():
        return None
    timeline = sorted((t, m) for t, m in history if t < moment)
    times = np.array([t for t, _ in timeline] + [moment])
    soil = profile.ranges["soilMoisture"]
    temperature = profile.ranges["temperature"]
    soil_series = np.array(
        [_value(m, "soilMoisture") for _, m in timeline] + [measures.get("soilMoisture")],
        dtype=float,
    )
    temp_series = np.array(
        [_value(m, "temperature") for _, m in timeline] + [measures.get("temperature")],
        dtype=float,
    )
    hours = np.array([np.nan if hour is None else float(hour)])
    return np.column_stack(
        [
            pos,
            _hour_features(hours, np.array([moment])),
            _slopes(times, soil_series, soil.max - soil.min)[-1:],
            _slopes(times, temp_series, temperature.max - temperature.min)[-1:],
        ]
    )


def camel_columns() -> list[str]:
    inverse = {v: k for k, v in STORE_NAMES.items()}
    return [inverse.get(c, c) for c in COLUMNS]


def _value(measures: dict[str, float | None], parameter: str) -> float:
    value = measures.get(parameter)
    return np.nan if value is None else float(value)


def to_store(measures: dict[str, float | None]) -> dict[str, float | None]:
    return {STORE_NAMES.get(name, name): value for name, value in measures.items()}
