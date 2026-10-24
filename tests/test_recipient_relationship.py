
import pandas as pd
import pytest

from src.recipient_relationship import detect_recipient_relationships


def make_data(amounts, receivers=None):
    if receivers is None:
        receivers = ["B"] * len(amounts)

    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(len(amounts))],
        "timestamp": pd.date_range(
            "2026-10-01", periods=len(amounts), freq="D"
        ),
        "sender_id": ["A"] * len(amounts),
        "receiver_id": receivers,
        "amount": amounts,
    })


def test_detects_new_recipient():
    data = make_data([100, 120], ["B", "C"])
    result = detect_recipient_relationships(data)

    assert result.iloc[0]["recipient_relationship_new"]
    assert result.iloc[1]["recipient_relationship_new"]


def test_detects_unusually_large_amount_for_familiar_recipient():
    data = make_data([100, 100, 100, 500])
    result = detect_recipient_relationships(data)

    assert result.iloc[-1]["recipient_prior_transaction_count"] == 3
    assert result.iloc[-1]["recipient_prior_average_amount"] == 100
    assert result.iloc[-1]["recipient_amount_deviation_flag"]


def test_does_not_flag_amount_without_enough_relationship_history():
    data = make_data([100, 500])
    result = detect_recipient_relationships(data)

    assert not result.iloc[-1]["recipient_amount_deviation_flag"]


def test_relationship_history_is_separate_for_each_sender():
    data = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "timestamp": pd.date_range(
            "2026-10-01", periods=4, freq="D"
        ),
        "sender_id": ["A", "A", "A", "C"],
        "receiver_id": ["B", "B", "B", "B"],
        "amount": [100, 100, 100, 500],
    })

    result = detect_recipient_relationships(data)
    last = result.iloc[-1]

    assert last["recipient_prior_transaction_count"] == 0
    assert last["recipient_relationship_new"]
    assert not last["recipient_amount_deviation_flag"]


def test_missing_required_column_raises_error():
    data = pd.DataFrame({
        "transaction_id": ["T1"],
        "sender_id": ["A"],
        "receiver_id": ["B"],
        "amount": [100],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        detect_recipient_relationships(data)


def test_empty_input_returns_expected_columns():
    data = pd.DataFrame(columns=[
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
    ])

    result = detect_recipient_relationships(data)

    assert result.empty
    assert "recipient_relationship_new" in result.columns
    assert "recipient_amount_deviation_flag" in result.columns
