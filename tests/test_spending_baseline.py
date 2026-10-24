
import pandas as pd

from src.spending_baseline import detect_spending_baseline


def test_flags_amount_far_above_personal_history():
    transactions = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(1, 8)],
        "sender_id": ["A"] * 7,
        "amount": [100, 100, 100, 100, 100, 100, 500],
    })

    result = detect_spending_baseline(transactions)

    final_transaction = result.iloc[-1]

    assert final_transaction["historical_transaction_count"] == 6
    assert final_transaction["historical_avg_amount"] == 100
    assert final_transaction["spending_change_flag"]


def test_does_not_flag_without_enough_history():
    transactions = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "sender_id": ["A", "A"],
        "amount": [100, 1000],
    })

    result = detect_spending_baseline(transactions)

    assert not result["spending_change_flag"].any()


def test_sender_histories_are_separate():
    transactions = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "sender_id": ["A", "A", "A", "A", "A", "A", "B"],
        "amount": [100, 100, 100, 100, 100, 500, 500],
    })

    result = detect_spending_baseline(transactions)
    transaction_a = result[result["transaction_id"] == "T6"].iloc[0]
    transaction_b = result[result["transaction_id"] == "T7"].iloc[0]

    assert transaction_a["spending_change_flag"]
    assert transaction_b["historical_transaction_count"] == 0
    assert not transaction_b["spending_change_flag"]


def test_timestamped_transactions_use_previous_history_only():
    transactions = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "sender_id": ["A"] * 7,
        "timestamp": pd.date_range(
            "2026-01-01", periods=7, freq="D"
        ),
        "amount": [100, 100, 100, 100, 100, 100, 500],
    })

    result = detect_spending_baseline(transactions)
    final_transaction = result.iloc[-1]

    assert final_transaction["historical_transaction_count"] == 6
    assert final_transaction["historical_avg_amount"] == 100
    assert final_transaction["spending_change_flag"]


def test_empty_input_returns_expected_columns():
    transactions = pd.DataFrame({
        "transaction_id": pd.Series(dtype=str),
        "sender_id": pd.Series(dtype=str),
        "amount": pd.Series(dtype=float),
    })

    result = detect_spending_baseline(transactions)

    assert "spending_change_flag" in result.columns
    assert len(result) == 0
