"""Base de conocimiento: rangos óptimos por especie en la escala de los sensores del dispositivo.

La conductividad se expresa en ppm con factor 700 (EC × 700) y la luz en lux relativos del
sensor del dispositivo (0 a 2000).
"""

from dataclasses import dataclass, field

PARAMETERS = ("temperature", "humidity", "brightness", "ph", "tds", "soilMoisture")

PHRASES = {
    "temperature": ("temperatura baja", "temperatura alta"),
    "humidity": ("aire seco", "aire muy húmedo"),
    "brightness": ("poca luz", "exceso de luz"),
    "ph": ("pH bajo", "pH alto"),
    "tds": ("pocos nutrientes", "exceso de nutrientes"),
    "soilMoisture": ("sustrato seco", "sustrato encharcado"),
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
        """0 dentro del rango; fuera, la distancia medida en tolerancias."""
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
            "temperature": Range(*temperature, tolerance=5.0, unit="°C", label="la temperatura"),
            "humidity": Range(*humidity, tolerance=15.0, unit="%", label="la humedad del aire"),
            "brightness": Range(*brightness, tolerance=300.0, unit="lux", label="la luz"),
            "ph": Range(*ph, tolerance=0.8, unit="pH", label="el pH"),
            "tds": Range(*tds, tolerance=400.0, unit="ppm", label="el nivel de nutrientes (TDS)"),
            "soilMoisture": Range(*soil, tolerance=15.0, unit="%", label="la humedad del sustrato"),
        },
    )


PROFILES: dict[str, CropProfile] = {
    profile.type: profile
    for profile in (
        _profile(
            "LETTUCE",
            "Lechuga",
            "Hoja de clima fresco; se espiga con calor y luz excesiva.",
            (15, 22),
            (50, 70),
            (300, 1400),
            (5.5, 6.5),
            (560, 840),
            (60, 80),
        ),
        _profile(
            "TOMATO",
            "Tomate",
            "Fruto exigente en luz, calor y nutrientes durante la floración.",
            (20, 28),
            (60, 80),
            (600, 1800),
            (5.5, 6.5),
            (1400, 2800),
            (55, 75),
        ),
        _profile(
            "STRAWBERRY",
            "Fresa",
            "Sensible al exceso de sales y a la humedad alta, que favorece hongos.",
            (18, 26),
            (60, 75),
            (500, 1600),
            (5.5, 6.2),
            (700, 980),
            (60, 80),
        ),
        _profile(
            "BASIL",
            "Albahaca",
            "Aromática de clima cálido; tolera ambientes secos.",
            (20, 30),
            (40, 60),
            (500, 1600),
            (5.5, 6.5),
            (700, 1120),
            (50, 70),
        ),
        _profile(
            "SPINACH",
            "Espinaca",
            "Hoja de clima fresco con alta demanda de nitrógeno.",
            (15, 24),
            (50, 70),
            (300, 1400),
            (6.0, 7.0),
            (1260, 1610),
            (60, 80),
        ),
        _profile(
            "PEPPER",
            "Pimentón",
            "Fruto de clima cálido; necesita luz intensa y pH estable.",
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
        raise KeyError(f"Tipo de cultivo desconocido: {crop_type}")
    return profile
