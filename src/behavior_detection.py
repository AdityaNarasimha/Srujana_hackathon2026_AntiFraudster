
import pandas as pd
from collections import deque


def detect_behavior(
    transactions: pd.DataFrame,
    rapid_window_minutes: int = 5,
    rapid_min_transactions: int = 3,
    min_history_transactions: int = 5,
) -> pd.DataFrame:
    """
    Detect rapid transfers and new recipients using past transactions.

    early_transfer_warning_flag:
        Live/causal early warning from the second transaction onward.

    rapid_transfer_flag:
        Live/causal signal. Flags from the third qualifying transfer onward
        by default.

    rapid_burst_review_flag:
        Retrospective signal. Flags all transactions belonging to a
        qualifying burst, including earlier transactions in that burst.

    These signals indicate patterns for review, not proof of fraud.
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
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if rapid_window_minutes <= 0:
        raise ValueError("rapid_window_minutes must be positive.")

    if rapid_min_transactions < 2:
        raise ValueError("rapid_min_transactions must be at least 2.")

    if min_history_transactions < 0:
        raise ValueError("min_history_transactions cannot be negative.")

    result = transactions.copy()
    result["early_transfer_warning_flag"] = False
    result["rapid_transfer_flag"] = False
    result["rapid_burst_review_flag"] = False
    result["new_recipient_flag"] = False
    result["behavior_reasons"] = ""

    if result.empty:
        return result

    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise"
    )
    result["amount"] = pd.to_numeric(
        result["amount"], errors="raise"
    )

    if result["timestamp"].isna().any():
        raise ValueError("Timestamps must not be missing.")

    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise ValueError("Amounts must be valid non-negative numbers.")

    if result[["sender_id", "receiver_id"]].isna().any().any():
        raise ValueError("Sender IDs and receiver IDs must not be missing.")

    if result["transaction_id"].isna().any():
        raise ValueError("Transaction IDs must not be missing.")

    result["_original_order"] = range(len(result))

    ordered = result.sort_values(
        ["timestamp", "_original_order"],
        kind="stable",
    )

    recent_times = {}
    seen_recipients = {}
    prior_transaction_count = {}
    window = pd.Timedelta(minutes=rapid_window_minutes)

    # Live detection: process one transaction at a time.
    # Transactions at the same timestamp are processed in stable input order.
    for idx, row in ordered.iterrows():
        sender = row["sender_id"]
        receiver = row["receiver_id"]
        timestamp = row["timestamp"]
        reasons = []

        times = recent_times.setdefault(sender, deque())

        # Remove transactions outside the rolling time window.
        while times and timestamp - times[0] > window:
            times.popleft()

        # Count earlier transactions plus the current transaction.
        observed_count = len(times) + 1

        if observed_count >= 2:
            result.at[idx, "early_transfer_warning_flag"] = True
            reasons.append(
                f"At least 2 transactions within "
                f"{rapid_window_minutes} minutes"
            )

        if observed_count >= rapid_min_transactions:
            result.at[idx, "rapid_transfer_flag"] = True
            reasons.append(
                f"At least {rapid_min_transactions} transactions "
                f"within {rapid_window_minutes} minutes"
            )

        times.append(timestamp)

        previous_recipients = seen_recipients.setdefault(sender, set())
        history_count = prior_transaction_count.get(sender, 0)

        if (
            history_count >= min_history_transactions
            and receiver not in previous_recipients
        ):
            result.at[idx, "new_recipient_flag"] = True
            reasons.append("Recipient not seen in prior sender history")

        previous_recipients.add(receiver)
        prior_transaction_count[sender] = history_count + 1
        result.at[idx, "behavior_reasons"] = "; ".join(reasons)

    # Retrospective investigation only; not suitable as a live prediction.
    for sender, group in ordered.groupby("sender_id", sort=False):
        group = group.sort_values(
            ["timestamp", "_original_order"],
            kind="stable",
        )

        timestamps = group["timestamp"].tolist()
        indices = group.index.tolist()
        left = 0

        for right in range(len(timestamps)):
            while (
                left < right
                and timestamps[right] - timestamps[left] > window
            ):
                left += 1

            if right - left + 1 >= rapid_min_transactions:
                for position in range(left, right + 1):
                    result.at[
                        indices[position], "rapid_burst_review_flag"
                    ] = True

    return result.drop(columns=["_original_order"])
