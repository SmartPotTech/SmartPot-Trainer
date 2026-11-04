"""Knowledge base: optimal species ranges on the device sensor scale.

Electrical conductivity is expressed in ppm with factor 700 (EC × 700) and light in relative lux
from the device sensor (0 to 2000).
"""

from dataclasses import dataclass, field

PARAMETERS = ("temperature", "humidity", "brightness", "ph", "tds", "soilMoisture")

PHRASES = {
    "temperature": ("low temperature", "high temperature"),
    "humidity": ("dry air", "very humid air"),
    "brightness": ("low light", "excess light"),
    "ph": ("low pH", "high pH"),
    "tds": ("low nutrients", "excess nutrients"),
    "soilMoisture": ("dry substrate", "waterlogged substrate"),
}


def phrase(parameter: str, status: str) -> str:
    low, high = PHRASES[parameter]
    return low if status == "LOW" else high


@dataclass(frozen=True)
class Range:
    min: float
    max: float
    tolerance: float
    unit: str
    label: str

    def deviation(self, value: float) -> float:
        """Returns 0 within the range; outside, the distance measured in tolerance units."""
        if value < self.min:
            return (self.min - value) / self.tolerance
        if value > self.max:
            return (value - self.max) / self.tolerance
        return 0.0


@dataclass(frozen=True)
class CropProfile:
    type: str
    name: str
    description: str
    ranges: dict[str, Range] = field(default_factory=dict)


def _profile(
    type_: str,
    name: str,
    description: str,
    temperature: tuple[float, float],
    humidity: tuple[float, float],
    brightness: tuple[float, float],
    ph: tuple[float, float],
    tds: tuple[float, float],
    soil: tuple[float, float],
) -> CropProfile:
    return CropProfile(
        type=type_,
        name=name,
        description=description,
        ranges={
            "temperature": Range(*temperature, tolerance=5.0, unit="°C", label="temperature"),
            "humidity": Range(*humidity, tolerance=15.0, unit="%", label="air humidity"),
            "brightness": Range(*brightness, tolerance=300.0, unit="lux", label="light"),
            "ph": Range(*ph, tolerance=0.8, unit="pH", label="pH"),
            "tds": Range(*tds, tolerance=400.0, unit="ppm", label="nutrient level (TDS)"),
            "soilMoisture": Range(*soil, tolerance=15.0, unit="%", label="substrate moisture"),
        },
    )


PROFILES: dict[str, CropProfile] = {
    profile.type: profile
    for profile in (
        _profile(
            "LETTUCE",
            "Lettuce",
            "Cool-weather leafy green; bolts under heat and excessive light.",
            (15, 22),
            (50, 70),
            (300, 1400),
            (5.5, 6.5),
            (560, 840),
            (60, 80),
        ),
        _profile(
            "TOMATO",
            "Tomato",
            "Fruiting crop demanding high light, warmth, and nutrients during flowering.",
            (20, 28),
            (60, 80),
            (600, 1800),
            (5.5, 6.5),
            (1400, 2800),
            (55, 75),
        ),
        _profile(
            "STRAWBERRY",
            "Strawberry",
            "Sensitive to excess salts and high humidity, which promotes fungal growth.",
            (18, 26),
            (60, 75),
            (500, 1600),
            (5.5, 6.2),
            (700, 980),
            (60, 80),
        ),
        _profile(
            "BASIL",
            "Basil",
            "Warm-weather aromatic herb; tolerates dry environments.",
            (20, 30),
            (40, 60),
            (500, 1600),
            (5.5, 6.5),
            (700, 1120),
            (50, 70),
        ),
        _profile(
            "SPINACH",
            "Spinach",
            "Cool-weather leafy green with high nitrogen demand.",
            (15, 24),
            (50, 70),
            (300, 1400),
            (6.0, 7.0),
            (1260, 1610),
            (60, 80),
        ),
        _profile(
            "PEPPER",
            "Bell pepper",
            "Warm-weather fruiting crop; requires intense light and stable pH.",
            (21, 29),
            (50, 70),
            (600, 1800),
            (5.8, 6.5),
            (1400, 2100),
            (55, 75),
        ),
    )
}


def get_profile(crop_type: str) -> CropProfile:
    profile = PROFILES.get(crop_type.upper())
    if profile is None:
        raise KeyError(f"Unknown crop type: {crop_type}")
    return profile
