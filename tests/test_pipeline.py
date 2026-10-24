
import pandas as pd
import pytest

from src.pipeline import analyze_transactions


def test_pipeline_combines_risk_scores():
    transactions = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "sender_id": ["A", "A"],
        "receiver_id": ["B", "C"],
        "amount": [100, 200],
    })

    customers = pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "customer_type": ["student", "salaried", "shopkeeper"],
        "expected_monthly_outflow": [1000, 5000, 10000],
        "usual_transaction_count": [10, 20, 30],
        "kyc_status": [
            "verified_demo",
            "verified_demo",
            "verified_demo",
        ],
    })

    result = analyze_transactions(transactions, customers)

    assert len(result) == 2
    assert "combined_risk_score" in result.columns
    assert "combined_risk_level" in result.columns
    assert result["combined_risk_score"].between(0, 100).all()


def test_pipeline_rejects_missing_transaction_columns():
    transactions = pd.DataFrame({
        "transaction_id": ["T1"],
        "amount": [100],
    })

    customers = pd.DataFrame({
        "customer_id": ["A"],
        "customer_type": ["student"],
        "expected_monthly_outflow": [1000],
        "usual_transaction_count": [10],
        "kyc_status": ["verified_demo"],
    })

    with pytest.raises(ValueError):
        analyze_transactions(transactions, customers)
