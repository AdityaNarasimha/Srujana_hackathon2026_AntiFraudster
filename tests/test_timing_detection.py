
import pandas as pd
import pytest

from src.timing_detection import detect_timing_anomalies


def make_transactions(hours):
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(len(hours))],
        "sender_id": ["A"] * len(hours),
        "timestamp": [
            pd.Timestamp("2026-01-01", tz="UTC")
            + pd.Timedelta(days=i, hours=hour)
            for i, hour in enumerate(hours)
        ],
    })


def test_flags_previously_unseen_hour_after_enough_history():
    transactions = make_transactions([10] * 10 + [2])

    result = detect_timing_anomalies(transactions)

    assert result.iloc[-1]["historical_timing_count"] == 10
    assert result.iloc[-1]["unusual_time_flag"]


def test_does_not_flag_an_observed_hour():
    transactions = make_transactions([10] * 11)

    result = detect_timing_anomalies(transactions)

    assert not result.iloc[-1]["unusual_time_flag"]


def test_does_not_flag_with_insufficient_history():
    transactions = make_transactions([10, 11, 2])

    result = detect_timing_anomalies(
        transactions,
        minimum_history=5,
    )

    assert not result.iloc[-1]["unusual_time_flag"]
    assert "Insufficient" in result.iloc[-1]["timing_reason"]


def test_sender_histories_are_separate():
    transactions = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(11)],
        "sender_id": ["A"] * 10 + ["B"],
        "timestamp": (
            [
                pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(days=i, hours=10)
                for i in range(10)
            ]
            + [pd.Timestamp("2026-02-01 02:00:00", tz="UTC")]
        ),
    })

    result = detect_timing_anomalies(transactions)

    sender_b = result[result["sender_id"] == "B"].iloc[0]

    assert sender_b["historical_timing_count"] == 0
    assert not sender_b["unusual_time_flag"]


def test_missing_required_column_raises_error():
    transactions = pd.DataFrame({
        "sender_id": ["A"],
    })

    with pytest.raises(ValueError):
        detect_timing_anomalies(transactions)


def test_empty_input_returns_expected_columns():
    transactions = pd.DataFrame({
        "sender_id": pd.Series(dtype=str),
        "timestamp": pd.Series(dtype=str),
    })

    result = detect_timing_anomalies(transactions)

    assert "unusual_time_flag" in result.columns
    assert result.empty
