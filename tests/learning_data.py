"""Synthetic time series modeling crop physics: substrate dries faster under heat and is watered when dropping
below the minimum; temperature and light follow the daily cycle."""

import math
import random

from app.learning.store import StoredReading

START = 1_790_000_000.0


def lettuce_series(
    crop_id: str,
    days: float = 3.0,
    step_minutes: float = 4.0,
    seed: int = 1,
    start: float = START,
) -> list[StoredReading]:
    rng = random.Random(seed)
    readings = []
    soil = 78.0
    steps = int(days * 24 * 60 / step_minutes)
    for i in range(steps):
        moment = start + i * step_minutes * 60
        hour = (moment / 3600 - 5) % 24
        daylight = max(0.0, math.sin((hour - 6) / 12 * math.pi))
        temperature = 16 + 9 * daylight + rng.gauss(0, 0.3)
        humidity = 68 - 14 * daylight + rng.gauss(0, 1)
        brightness = max(0.0, 60 + 1300 * daylight + rng.gauss(0, 20))
        # Más calor y más luz secan más rápido; al bajar de 58 % alguien riega.
        soil -= (0.12 + 0.35 * daylight) * step_minutes / 4
        if soil < 58:
            soil = 80.0
        readings.append(
            StoredReading(
                crop_id=crop_id,
                crop_type="LETTUCE",
                measured_at=moment,
                local_hour=int(hour),
                values={
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "brightness": round(brightness),
                    "ph": round(6.0 + rng.gauss(0, 0.05), 2),
                    "tds": round(700 + rng.gauss(0, 10)),
                    "atmosphere": 1012.0,
                    "soil_moisture": round(soil + rng.gauss(0, 0.3), 1),
                },
            )
        )
    return readings
