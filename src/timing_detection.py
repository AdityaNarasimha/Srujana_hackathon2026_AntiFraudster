
import pandas as pd


def detect_timing_anomalies(
    transactions: pd.DataFrame,
    minimum_history: int = 10,
    unusual_hour_minimum_count: int = 2,
) -> pd.DataFrame:
    """
    Flag transactions during hours that differ from the sender's
    previously observed activity.

    Uses previous transactions only. This is a review signal,
    not proof of fraud.
    """

    required = {"sender_id", "timestamp"}
    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1.")

    if unusual_hour_minimum_count < 1:
        raise ValueError(
            "unusual_hour_minimum_count must be at least 1."
        )

    result = transactions.copy()
    result["unusual_time_flag"] = False
    result["historical_timing_count"] = 0
    result["historical_active_hours"] = ""
    result["timing_reason"] = ""

    if result.empty:
        return result

    result["_original_order"] = range(len(result))
    result["_parsed_timestamp"] = pd.to_datetime(
        result["timestamp"], errors="raise", utc=True
    )

    if result["_parsed_timestamp"].isna().any():
        raise ValueError("Timestamps must not be missing.")

    ordered = result.sort_values(
        ["_parsed_timestamp", "_original_order"],
        kind="mergesort",
    )

    # Store past activity-hour counts separately for each sender.
    sender_hour_counts = {}
    sender_history_counts = {}

    for idx, row in ordered.iterrows():
        sender = row["sender_id"]
        timestamp = row["_parsed_timestamp"]
        current_hour = timestamp.hour

        hour_counts = sender_hour_counts.setdefault(sender, {})
        history_count = sender_history_counts.get(sender, 0)

        result.at[idx, "historical_timing_count"] = history_count

        active_hours = sorted(
            hour for hour, count in hour_counts.items()
            if count >= unusual_hour_minimum_count
        )

        result.at[idx, "historical_active_hours"] = ", ".join(
            f"{hour:02d}:00" for hour in active_hours
        )

        # Require enough past activity and a meaningful observed pattern.
        if history_count >= minimum_history:
            current_hour_count = hour_counts.get(current_hour, 0)

            # A previously unseen hour is a review signal only when
            # the sender already has sufficient historical activity.
            if current_hour_count == 0:
                result.at[idx, "unusual_time_flag"] = True
                result.at[idx, "timing_reason"] = (
                    f"No previous transactions observed during "
                    f"hour {current_hour:02d}:00"
                )
            else:
                result.at[idx, "timing_reason"] = (
                    "Transaction hour observed in prior history"
                )
        else:
            result.at[idx, "timing_reason"] = (
                "Insufficient personal timing history"
            )

        # Update history only after evaluating this transaction.
        hour_counts[current_hour] = (
            hour_counts.get(current_hour, 0) + 1
        )
        sender_history_counts[sender] = history_count + 1

    result = result.drop(
        columns=["_original_order", "_parsed_timestamp"]
    )

    return result
