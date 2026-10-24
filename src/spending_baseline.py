
import pandas as pd


def detect_spending_baseline(
    transactions: pd.DataFrame,
    minimum_history: int = 5,
    deviation_multiplier: float = 3.0,
    minimum_baseline_amount: float = 0.0,
) -> pd.DataFrame:
    """
    Compare each transaction with the sender's previous spending.

    Only earlier transactions contribute to the baseline, preventing
    the current transaction from influencing its own expected amount.

    This is a review signal, not a fraud probability or proof of fraud.
    """

    required = {"sender_id", "amount"}
    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1.")

    if deviation_multiplier <= 1:
        raise ValueError("deviation_multiplier must be greater than 1.")

    if minimum_baseline_amount < 0:
        raise ValueError("minimum_baseline_amount cannot be negative.")

    result = transactions.copy()
    result["amount"] = pd.to_numeric(
        result["amount"], errors="coerce"
    )

    if result["amount"].isna().any() or (
        result["amount"] < 0
    ).any():
        raise ValueError(
            "Transaction amounts must be valid non-negative numbers."
        )

    # Keep the original order for returning results.
    result["_original_order"] = range(len(result))

    has_timestamp = "timestamp" in result.columns

    if has_timestamp:
        result["_parsed_timestamp"] = pd.to_datetime(
            result["timestamp"], errors="coerce", utc=True
        )
        # If timestamps cannot be parsed, preserve original order.
        result["_sort_timestamp"] = result["_parsed_timestamp"]
    else:
        result["_parsed_timestamp"] = pd.NaT
        result["_sort_timestamp"] = pd.NaT

    result = result.sort_values(
        ["sender_id", "_sort_timestamp", "_original_order"],
        kind="mergesort",
        na_position="last",
    )

    historical_counts = []
    historical_averages = []
    deviation_ratios = []
    flags = []
    reasons = []

    # Process each sender independently.
    for _, group in result.groupby(
        "sender_id", sort=False, dropna=False
    ):
        previous_amounts = []
        previous_timestamp = None

        for _, row in group.iterrows():
            amount = float(row["amount"])
            timestamp = row["_parsed_timestamp"]

            # A timestamped transaction must not use transactions
            # with unknown timestamps as its history.
            if pd.isna(timestamp):
                usable_history = (
                    previous_amounts
                    if not has_timestamp
                    else []
                )
            else:
                usable_history = previous_amounts

            count = len(usable_history)

            if count:
                average = sum(usable_history) / count
            else:
                average = float("nan")

            if pd.notna(average) and average > 0:
                ratio = amount / average
            else:
                ratio = float("nan")

            enough_history = count >= minimum_history
            flagged = (
                enough_history
                and pd.notna(average)
                and average >= minimum_baseline_amount
                and average > 0
                and amount > average * deviation_multiplier
            )

            if not enough_history:
                reason = "Insufficient personal spending history"
            elif pd.isna(average) or average <= 0:
                reason = "No usable positive spending baseline"
            elif flagged:
                reason = (
                    "Amount substantially exceeds the sender's "
                    "historical average"
                )
            else:
                reason = "No spending-baseline deviation detected"

            historical_counts.append((row["_original_order"], count))
            historical_averages.append(
                (row["_original_order"], average)
            )
            deviation_ratios.append((row["_original_order"], ratio))
            flags.append((row["_original_order"], bool(flagged)))
            reasons.append((row["_original_order"], reason))

            # Only add this transaction to history after evaluating it.
            # With timestamps, unknown-time rows are not used as history.
            if pd.notna(timestamp):
                previous_amounts.append(amount)
                previous_timestamp = timestamp
            elif not has_timestamp:
                previous_amounts.append(amount)

    def restore_order(pairs, column):
        values = dict(pairs)
        return result["_original_order"].map(values).rename(column)

    result["historical_transaction_count"] = restore_order(
        historical_counts, "historical_transaction_count"
    ).astype(int)

    result["historical_avg_amount"] = restore_order(
        historical_averages, "historical_avg_amount"
    )

    result["spending_deviation_ratio"] = restore_order(
        deviation_ratios, "spending_deviation_ratio"
    )

    result["spending_change_flag"] = restore_order(
        flags, "spending_change_flag"
    ).astype(bool)

    result["spending_change_reason"] = restore_order(
        reasons, "spending_change_reason"
    )

    result["spending_baseline_review"] = (
        result["spending_change_flag"]
    )

    return result.drop(
        columns=[
            "_original_order",
            "_parsed_timestamp",
            "_sort_timestamp",
        ]
    )
