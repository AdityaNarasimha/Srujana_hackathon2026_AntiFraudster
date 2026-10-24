
import pandas as pd

from src.location_detection import detect_location_anomalies


def test_impossible_travel_is_flagged():
    data = pd.DataFrame({
        "timestamp": [
            "2026-09-01 10:00:00",
            "2026-09-01 10:05:00",
        ],
        "sender_id": ["ACC-001", "ACC-001"],
        "location_id": ["LOC-001", "LOC-005"],
    })

    result = detect_location_anomalies(data)

    assert not result.iloc[0]["location_anomaly_flag"]
    assert result.iloc[1]["location_anomaly_flag"]


def test_same_location_is_not_flagged():
    data = pd.DataFrame({
        "timestamp": [
            "2026-09-01 10:00:00",
            "2026-09-01 10:01:00",
        ],
        "sender_id": ["ACC-001", "ACC-001"],
        "location_id": ["LOC-001", "LOC-001"],
    })

    result = detect_location_anomalies(data)

    assert not result["location_anomaly_flag"].any()


def test_missing_location_is_not_flagged():
    data = pd.DataFrame({
        "timestamp": [
            "2026-09-01 10:00:00",
            "2026-09-01 10:05:00",
        ],
        "sender_id": ["ACC-001", "ACC-001"],
        "location_id": ["LOC-001", None],
    })

    result = detect_location_anomalies(data)

    assert not result["location_anomaly_flag"].any()
