import pandas as pd
import pytest

from src import train_evaluate_ml as ml


def make_transactions():
    """Small chronological dataset for fast, deterministic tests."""
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "timestamp": pd.to_datetime([
            "2026-09-01 10:00:00",
            "2026-09-01 10:02:00",
            "2026-09-01 10:04:00",
            "2026-09-01 23:00:00",
        ]),
        "sender_id": ["A", "A", "A", "B"],
        "receiver_id": ["R1", "R2", "R1", "R3"],
        "amount": [100.0, 200.0, 50.0, 80.0],
        "location_id": ["L1", "L1", "L2", "L1"],
        "is_fraud": [0, 1, 0, 0],
    })


def test_historical_count_uses_only_previous_transactions():
    result = ml.add_historical_features(make_transactions())

    assert result["sender_history_count"].tolist() == [0, 1, 2, 0]


def test_historical_average_and_max_exclude_current_transaction():
    result = ml.add_historical_features(make_transactions())

    assert pd.isna(result.loc[0, "sender_history_avg_amount"])
    assert pd.isna(result.loc[0, "sender_history_max_amount"])

    assert result.loc[1, "sender_history_avg_amount"] == 100.0
    assert result.loc[1, "sender_history_max_amount"] == 100.0

    assert result.loc[2, "sender_history_avg_amount"] == 150.0
    assert result.loc[2, "sender_history_max_amount"] == 200.0


def test_seconds_since_previous_transaction():
    result = ml.add_historical_features(make_transactions())

    assert pd.isna(result.loc[0, "seconds_since_previous"])
    assert result.loc[1, "seconds_since_previous"] == 120.0
    assert result.loc[2, "seconds_since_previous"] == 120.0
    assert pd.isna(result.loc[3, "seconds_since_previous"])


def test_known_recipient_uses_sender_history():
    result = ml.add_historical_features(make_transactions())

    assert result["known_recipient"].tolist() == [0, 0, 1, 0]


def test_amount_history_features():
    result = ml.add_historical_features(make_transactions())

    assert result.loc[1, "amount_to_history_ratio"] == 2.0
    assert result.loc[1, "amount_above_history_max"] == 1

    assert result.loc[2, "amount_to_history_ratio"] == pytest.approx(1 / 3)
    assert result.loc[2, "amount_above_history_max"] == 0


def test_time_features():
    result = ml.add_historical_features(make_transactions())

    assert result["hour"].tolist() == [10, 10, 10, 23]
    assert result["is_night"].tolist() == [0, 0, 0, 1]


def test_feature_generation_does_not_change_input():
    data = make_transactions()
    original_columns = data.columns.tolist()

    ml.add_historical_features(data)

    assert data.columns.tolist() == original_columns
    assert "sender_history_count" not in data.columns


def test_load_data_sorts_chronologically(tmp_path, monkeypatch):
    data = make_transactions().iloc[[2, 0, 3, 1]]
    csv_path = tmp_path / "transactions.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(ml, "DATA_PATH", csv_path)

    result = ml.load_data()

    assert result["transaction_id"].tolist() == ["T1", "T2", "T3", "T4"]


def test_load_data_rejects_missing_columns(tmp_path, monkeypatch):
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({"transaction_id": ["T1"]}).to_csv(csv_path, index=False)

    monkeypatch.setattr(ml, "DATA_PATH", csv_path)

    with pytest.raises(ValueError, match="Missing columns"):
        ml.load_data()


def test_load_data_rejects_invalid_fraud_labels(tmp_path, monkeypatch):
    data = make_transactions()
    data["is_fraud"] = [0, 1, 2, 0]

    csv_path = tmp_path / "transactions.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(ml, "DATA_PATH", csv_path)

    with pytest.raises(ValueError, match="only 0 and 1"):
        ml.load_data()
