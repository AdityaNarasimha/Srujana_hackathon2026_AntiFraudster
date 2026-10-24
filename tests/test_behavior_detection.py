
import pandas as pd
import pytest

from src.behavior_detection import detect_behavior


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "timestamp": [
            "2026-10-01 10:00:00",
            "2026-10-01 10:01:00",
            "2026-10-01 10:02:00",
            "2026-10-01 10:03:00",
            "2026-10-01 10:04:00",
            "2026-10-01 10:20:00",
        ],
        "sender_id": ["A", "A", "A", "A", "A", "A"],
        "receiver_id": ["B", "B", "B", "C", "C", "D"],
        "amount": [100, 120, 90, 150, 130, 200],
    })


def test_detects_rapid_transfers(sample_transactions):
    result = detect_behavior(
        sample_transactions,
        min_history_transactions=0,
    )

    # The third transaction reaches the three-transfer threshold.
    assert result.loc[2, "rapid_transfer_flag"]

    # The fourth and fifth are also inside the five-minute window.
    assert result.loc[3, "rapid_transfer_flag"]
    assert result.loc[4, "rapid_transfer_flag"]


def test_does_not_flag_slow_transactions():
    data = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "timestamp": ["2026-10-01 10:00:00", "2026-10-01 10:30:00"],
        "sender_id": ["A", "A"],
        "receiver_id": ["B", "C"],
        "amount": [100, 200],
    })

    result = detect_behavior(data)

    assert not result["rapid_transfer_flag"].any()


def test_new_recipient_requires_enough_history(sample_transactions):
    result = detect_behavior(
        sample_transactions,
        min_history_transactions=3,
    )

    # T4 is the fourth transaction and introduces recipient C.
    assert result.loc[3, "new_recipient_flag"]

    # Recipient B was seen previously.
    assert not result.loc[4, "new_recipient_flag"]


def test_missing_required_column_raises_error():
    data = pd.DataFrame({
        "transaction_id": ["T1"],
        "sender_id": ["A"],
        "receiver_id": ["B"],
        "amount": [100],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        detect_behavior(data)


def test_empty_input_returns_empty_result():
    data = pd.DataFrame(columns=[
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
    ])

    result = detect_behavior(data)

    assert result.empty
    assert "rapid_transfer_flag" in result.columns
    assert "new_recipient_flag" in result.columns
