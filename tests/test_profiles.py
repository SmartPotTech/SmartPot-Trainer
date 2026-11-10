# Comprobación de paridad agronómica entre SmartPot-Trainer y SmartPot-AI
from app.knowledge.profiles import PROFILES, get_profile


def test_all_six_canonical_profiles_exist():
    # Verifica que existan exactamente las 6 especies soportadas
    assert len(PROFILES) == 6
    assert set(PROFILES.keys()) == {
        "LETTUCE",
        "TOMATO",
        "STRAWBERRY",
        "BASIL",
        "SPINACH",
        "PEPPER",
    }


def test_canonical_descriptions_match_smartpot_ai():
    # Verifica que las descripciones coincidan exactamente con profiles.rs
    expected_descriptions = {
        "LETTUCE": "Cool-weather leafy green; bolts under excessive heat and light.",
        "TOMATO": "Fruit crop demanding in light, warmth, and nutrients during flowering.",
        "STRAWBERRY": "Sensitive to excess salts and high humidity, which promotes fungal growth.",
        "BASIL": "Warm-weather aromatic herb; tolerates dry environments.",
        "SPINACH": "Cool-weather leafy green with high nitrogen demand.",
        "PEPPER": "Warm-weather fruit crop; requires intense light and stable pH.",
    }
    for crop_type, expected_desc in expected_descriptions.items():
        profile = get_profile(crop_type)
        assert profile.description == expected_desc


def test_canonical_ranges_match_smartpot_ai():
    # Verifica los rangos agronómicos de las 6 especies
    lettuce = get_profile("LETTUCE")
    assert (lettuce.ranges["temperature"].min, lettuce.ranges["temperature"].max) == (15, 22)
    assert (lettuce.ranges["humidity"].min, lettuce.ranges["humidity"].max) == (50, 70)
    assert (lettuce.ranges["brightness"].min, lettuce.ranges["brightness"].max) == (300, 1400)
    assert (lettuce.ranges["ph"].min, lettuce.ranges["ph"].max) == (5.5, 6.5)
    assert (lettuce.ranges["tds"].min, lettuce.ranges["tds"].max) == (560, 840)
    assert (lettuce.ranges["soilMoisture"].min, lettuce.ranges["soilMoisture"].max) == (60, 80)

    tomato = get_profile("TOMATO")
    assert (tomato.ranges["temperature"].min, tomato.ranges["temperature"].max) == (20, 28)
    assert (tomato.ranges["humidity"].min, tomato.ranges["humidity"].max) == (60, 80)
    assert (tomato.ranges["brightness"].min, tomato.ranges["brightness"].max) == (600, 1800)
    assert (tomato.ranges["ph"].min, tomato.ranges["ph"].max) == (5.5, 6.5)
    assert (tomato.ranges["tds"].min, tomato.ranges["tds"].max) == (1400, 2800)
    assert (tomato.ranges["soilMoisture"].min, tomato.ranges["soilMoisture"].max) == (55, 75)

    strawberry = get_profile("STRAWBERRY")
    assert (strawberry.ranges["temperature"].min, strawberry.ranges["temperature"].max) == (18, 26)
    assert (strawberry.ranges["humidity"].min, strawberry.ranges["humidity"].max) == (60, 75)
    assert (strawberry.ranges["brightness"].min, strawberry.ranges["brightness"].max) == (500, 1600)
    assert (strawberry.ranges["ph"].min, strawberry.ranges["ph"].max) == (5.5, 6.2)
    assert (strawberry.ranges["tds"].min, strawberry.ranges["tds"].max) == (700, 980)
    assert (strawberry.ranges["soilMoisture"].min, strawberry.ranges["soilMoisture"].max) == (60, 80)

    basil = get_profile("BASIL")
    assert (basil.ranges["temperature"].min, basil.ranges["temperature"].max) == (20, 30)
    assert (basil.ranges["humidity"].min, basil.ranges["humidity"].max) == (40, 60)
    assert (basil.ranges["brightness"].min, basil.ranges["brightness"].max) == (500, 1600)
    assert (basil.ranges["ph"].min, basil.ranges["ph"].max) == (5.5, 6.5)
    assert (basil.ranges["tds"].min, basil.ranges["tds"].max) == (700, 1120)
    assert (basil.ranges["soilMoisture"].min, basil.ranges["soilMoisture"].max) == (50, 70)

    spinach = get_profile("SPINACH")
    assert (spinach.ranges["temperature"].min, spinach.ranges["temperature"].max) == (15, 24)
    assert (spinach.ranges["humidity"].min, spinach.ranges["humidity"].max) == (50, 70)
    assert (spinach.ranges["brightness"].min, spinach.ranges["brightness"].max) == (300, 1400)
    assert (spinach.ranges["ph"].min, spinach.ranges["ph"].max) == (6.0, 7.0)
    assert (spinach.ranges["tds"].min, spinach.ranges["tds"].max) == (1260, 1610)
    assert (spinach.ranges["soilMoisture"].min, spinach.ranges["soilMoisture"].max) == (60, 80)

    pepper = get_profile("PEPPER")
    assert (pepper.ranges["temperature"].min, pepper.ranges["temperature"].max) == (21, 29)
    assert (pepper.ranges["humidity"].min, pepper.ranges["humidity"].max) == (50, 70)
    assert (pepper.ranges["brightness"].min, pepper.ranges["brightness"].max) == (600, 1800)
    assert (pepper.ranges["ph"].min, pepper.ranges["ph"].max) == (5.8, 6.5)
    assert (pepper.ranges["tds"].min, pepper.ranges["tds"].max) == (1400, 2100)
    assert (pepper.ranges["soilMoisture"].min, pepper.ranges["soilMoisture"].max) == (55, 75)
