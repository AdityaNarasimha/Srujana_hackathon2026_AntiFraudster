
import pandas as pd


def detect_behavior(
    transactions: pd.DataFrame,
    rapid_window_minutes: int = 5,
    rapid_min_transactions: int = 3,
    min_history_transactions: int = 5,
) -> pd.DataFrame:
    """
    Detect rapid transfers and new recipients using past transactions only.

    Required columns:
        transaction_id, timestamp, sender_id, receiver_id, amount

    This module generates review signals, not proof of fraud.
    """

    # 1. Validate required columns.
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

    if rapid_window_minutes <= 0:
        raise ValueError("rapid_window_minutes must be positive.")

    if rapid_min_transactions < 2:
        raise ValueError("rapid_min_transactions must be at least 2.")

    if min_history_transactions < 0:
        raise ValueError("min_history_transactions cannot be negative.")

    # 2. Prepare the output.
    result = transactions.copy()

    result["rapid_transfer_flag"] = False
    result["new_recipient_flag"] = False
    result["behavior_reasons"] = ""

    if result.empty:
        return result

    # 3. Validate transaction values.
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise"
    )

    result["amount"] = pd.to_numeric(
        result["amount"], errors="raise"
    )

    if result["timestamp"].isna().any():
        raise ValueError("Timestamps must not be missing.")

    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise ValueError(
            "Amounts must be valid non-negative numbers."
        )

    if result[["sender_id", "receiver_id"]].isna().any().any():
        raise ValueError(
            "Sender IDs and receiver IDs must not be missing."
        )

    if result["transaction_id"].isna().any():
        raise ValueError("Transaction IDs must not be missing.")

    # 4. Process transactions in chronological order.
    result["_original_order"] = range(len(result))

    ordered = result.sort_values(
        ["timestamp", "_original_order"],
        kind="stable",
    )

    # Keep each sender's recent timestamps and recipient history.
    recent_times = {}
    seen_recipients = {}
    prior_transaction_count = {}

    window = pd.Timedelta(minutes=rapid_window_minutes)

    for idx, row in ordered.iterrows():
        sender = row["sender_id"]
        receiver = row["receiver_id"]
        timestamp = row["timestamp"]

        reasons = []

        # 5. Detect rapid transfers.
        times = recent_times.setdefault(sender, [])

        # Keep only transactions within the rolling time window.
        times = [
            previous_time
            for previous_time in times
            if timestamp - previous_time <= window
        ]

        times.append(timestamp)
        recent_times[sender] = times

        if len(times) >= rapid_min_transactions:
            result.at[idx, "rapid_transfer_flag"] = True
            reasons.append(
                f"At least {rapid_min_transactions} transactions "
                f"within {rapid_window_minutes} minutes"
            )

        # 6. Detect new recipients using prior history only.
        previous_recipients = seen_recipients.setdefault(sender, set())

        history_count = prior_transaction_count.get(sender, 0)

        if (
            history_count >= min_history_transactions
            and receiver not in previous_recipients
        ):
            result.at[idx, "new_recipient_flag"] = True
            reasons.append(
                "Recipient not seen in prior sender history"
            )

        # Update history AFTER assessing the current transaction.
        previous_recipients.add(receiver)
        prior_transaction_count[sender] = history_count + 1

        result.at[idx, "behavior_reasons"] = "; ".join(reasons)

    # 7. Restore the original row order.
    result = result.drop(columns=["_original_order"])

    return result
