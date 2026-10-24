
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_fraud(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potentially suspicious transactions using simple rules
    and Isolation Forest anomaly detection.

    Required columns:
        transaction_id, sender_id, receiver_id, amount
    """
    required = {
        "transaction_id",
        "sender_id",
        "receiver_id",
        "amount",
    }

    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if transactions.empty:
        result = transactions.copy()
        result["risk_score"] = pd.Series(dtype=float)
        result["risk_level"] = pd.Series(dtype=str)
        result["reasons"] = pd.Series(dtype=str)
        return result

    result = transactions.copy()
    amounts = pd.to_numeric(result["amount"], errors="coerce")

    if amounts.isna().any() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be valid non-negative numbers.")

    # Rule 1: unusually large amounts compared with the dataset.
    median_amount = amounts.median()
    amount_threshold = max(median_amount * 5, amounts.quantile(0.99))
    large_amount = amounts > amount_threshold

    # Rule 2: unusually frequent transfers from one sender.
    sender_counts = result.groupby("sender_id")["transaction_id"].transform("count")
    frequent_sender = sender_counts > sender_counts.quantile(0.95)

    # Rule 3: anomaly detection using amount and sender transaction frequency.
    features = pd.DataFrame({
        "amount": amounts,
        "sender_transaction_count": sender_counts,
    })

    anomaly = pd.Series(False, index=result.index)

    if len(result) >= 20 and features.nunique().min() > 1:
        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        )
        predictions = model.fit_predict(features)
        anomaly = pd.Series(predictions == -1, index=result.index)

    result["risk_score"] = (
        large_amount.astype(int) * 35
        + frequent_sender.astype(int) * 20
        + anomaly.astype(int) * 35
    ).clip(upper=100)

    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-1, 19, 49, 79, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    result["reasons"] = [
        "; ".join(reason for condition, reason in [
            (bool(large), "Amount unusually large"),
            (bool(frequent), "Sender has unusually frequent transactions"),
            (bool(outlier), "Anomaly detection flagged this transaction"),
        ] if condition) or "No implemented rule triggered"
        for large, frequent, outlier in zip(
            large_amount, frequent_sender, anomaly
        )
    ]

    return result
