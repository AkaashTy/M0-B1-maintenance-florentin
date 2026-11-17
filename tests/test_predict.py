from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
import pytest

def test_predict_returns_200() -> None:
    payload = {
        "age_machine_jours": 1500,
        "derniere_maintenance_jours": 45,
        "nb_incidents_3_mois": 2,
        "pression_moyenne": 7.8,
        "temperature_moyenne": 68.5,
        "type_machine": "compresseur",
        "vibration_moyenne": 3.2
        }
    """L'endpoint /predict doit répondre 200 OK avec le modèle chargé."""
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["criticite"] in {"basse","moyenne","haute"}
        assert sum(body["probabilites"].values()) == pytest.approx(1) 


TYPES_VALIDES = {"pompe", "compresseur", "convoyeur", "presse", "four"}
def test_predict_typeMachine_invalid() -> None:
    payload = {
        "age_machine_jours": 1500,
        "derniere_maintenance_jours": 45,
        "nb_incidents_3_mois": 2,
        "pression_moyenne": 7.8,
        "temperature_moyenne": 68.5,
        "type_machine": "INCONNU",
        "vibration_moyenne": 3.2
        }
    assert payload["type_machine"] not in TYPES_VALIDES
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        body = response.json()


@pytest.mark.parametrize("type_machine", [
"pompe", "compresseur", "convoyeur", "presse", "four",
])
def test_predict_type_machine_valid_param(type_machine: str) -> None:
    payload = {
        "age_machine_jours": 1500,
        "derniere_maintenance_jours": 45,
        "nb_incidents_3_mois": 2,
        "pression_moyenne": 7.8,
        "temperature_moyenne": 68.5,
        "type_machine": type_machine,
        "vibration_moyenne": 3.2
        }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert sum(body["probabilites"].values()) == pytest.approx(1) 

