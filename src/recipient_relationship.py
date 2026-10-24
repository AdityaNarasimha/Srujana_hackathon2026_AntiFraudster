
import pandas as pd


def detect_recipient_relationships(
    transactions: pd.DataFrame,
    minimum_history: int = 3,
    unusual_amount_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Analyze each sender-recipient relationship using previous
    transactions only.

    Produces explainable review signals, not proof of fraud.
    """

    required = {
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
    }
    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1.")

    if unusual_amount_multiplier <= 1:
        raise ValueError(
            "unusual_amount_multiplier must be greater than 1."
        )

    result = transactions.copy()
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")

    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise ValueError(
            "Amounts must be valid non-negative numbers."
        )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise", utc=True
    )

    if result["timestamp"].isna().any():
        raise ValueError("Timestamps must not be missing.")

    if result[
        ["transaction_id", "sender_id", "receiver_id"]
    ].isna().any().any():
        raise ValueError(
            "Transaction IDs, sender IDs and receiver IDs must not be missing."
        )

    result["_original_order"] = range(len(result))
    ordered = result.sort_values(
        ["timestamp", "_original_order"],
        kind="mergesort",
    )

    relationship_history = {}
    output = {}

    for idx, row in ordered.iterrows():
        sender = row["sender_id"]
        receiver = row["receiver_id"]
        amount = float(row["amount"])
        key = (sender, receiver)

        previous = relationship_history.setdefault(key, [])
        count = len(previous)

        average_amount = (
            sum(previous) / count if count else float("nan")
        )
        total_amount = sum(previous)

        new_recipient = count == 0
        unusual_amount = (
            count >= minimum_history
            and average_amount > 0
            and amount > average_amount * unusual_amount_multiplier
        )

        reasons = []
        if new_recipient:
            reasons.append("No previous sender-recipient transactions")
        if unusual_amount:
            reasons.append(
                "Amount substantially exceeds prior transfers "
                "to this recipient"
            )

        output[idx] = {
            "recipient_prior_transaction_count": count,
            "recipient_prior_total_amount": total_amount,
            "recipient_prior_average_amount": average_amount,
            "recipient_relationship_new": new_recipient,
            "recipient_amount_deviation_flag": unusual_amount,
            "recipient_relationship_reason": "; ".join(reasons),
        }

        # Update relationship history only after evaluating this row.
        previous.append(amount)

    for column in [
        "recipient_prior_transaction_count",
        "recipient_prior_total_amount",
        "recipient_prior_average_amount",
        "recipient_relationship_new",
        "recipient_amount_deviation_flag",
        "recipient_relationship_reason",
    ]:
        result[column] = pd.Series({
            idx: values[column] for idx, values in output.items()
        }).reindex(result.index)

    result["recipient_prior_transaction_count"] = (
        result["recipient_prior_transaction_count"].astype(int)
    )
    result["recipient_relationship_new"] = (
        result["recipient_relationship_new"].astype(bool)
    )
    result["recipient_amount_deviation_flag"] = (
        result["recipient_amount_deviation_flag"].astype(bool)
    )

    return result.drop(columns=["_original_order"])
