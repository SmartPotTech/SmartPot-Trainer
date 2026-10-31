def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert data["service"] == "smartpot-trainer"


def test_auth_protection(client):
    res = client.get("/v1/learning/status")
    assert res.status_code == 401

    res = client.get("/v1/learning/status", headers={"Authorization": "Bearer bad-token"})
    assert res.status_code == 401


def test_learning_endpoints(client, auth):
    # Status inicial
    res = client.get("/v1/learning/status", headers=auth)
    assert res.status_code == 200
    assert "cropTypes" in res.json()

    # Ingesta
    payload = {
        "readings": [
            {
                "cropId": "crop-api-test",
                "cropType": "LETTUCE",
                "measuredAt": "2026-10-31T12:00:00Z",
                "localHour": 12,
                "measures": {
                    "temperature": 21.5,
                    "humidity": 62.0,
                    "brightness": 850.0,
                    "ph": 6.1,
                    "tds": 720.0,
                    "atmosphere": 1013.0,
                    "soilMoisture": 72.0,
                },
            }
        ]
    }
    res = client.post("/v1/learning/readings", json=payload, headers=auth)
    assert res.status_code == 200
    data = res.json()
    assert data["received"] == 1
    assert data["stored"] == 1

    # Olvido de cultivo
    res = client.delete("/v1/learning/crops/crop-api-test", headers=auth)
    assert res.status_code == 200
    assert res.json()["deleted"] >= 1

    # Poda
    res = client.post("/v1/learning/prune", headers=auth)
    assert res.status_code == 200
    assert "removed" in res.json()


def test_validation_error(client, auth):
    # Tipo de cultivo inexistente
    payload = {
        "readings": [
            {
                "cropId": "crop-bad",
                "cropType": "BANANA_CROP_UNKNOWN",
                "measuredAt": "2026-10-31T12:00:00Z",
                "measures": {},
            }
        ]
    }
    res = client.post("/v1/learning/readings", json=payload, headers=auth)
    assert res.status_code == 422
