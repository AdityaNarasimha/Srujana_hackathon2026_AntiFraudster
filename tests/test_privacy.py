
import pandas as pd

from src.privacy import protect_transactions


def test_removes_extra_columns_and_masks_ids():
    df = pd.DataFrame({
        "transaction_id": ["TX-1"],
        "timestamp": ["2026-10-01 12:00:00"],
        "sender_id": ["ACCOUNT-123"],
        "receiver_id": ["ACCOUNT-456"],
        "amount": [500],
        "customer_name": ["Example Customer"],
        "is_fraud": [0],
    })

    protected, removed = protect_transactions(df)

    assert "customer_name" in removed
    assert "customer_name" not in protected.columns
    assert protected.loc[0, "sender_id"] != "ACCOUNT-123"
    assert protected.loc[0, "receiver_id"] != "ACCOUNT-456"
    assert protected.loc[0, "amount"] == 500


def test_missing_required_column_is_rejected():
    df = pd.DataFrame({
        "timestamp": ["2026-10-01 12:00:00"],
        "sender_id": ["A"],
        "receiver_id": ["B"],
    })

    try:
        protect_transactions(df)
    except ValueError as error:
        assert "amount" in str(error)
    else:
        raise AssertionError("Expected invalid data to be rejected.")
